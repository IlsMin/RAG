#include "KnowledgeBase.h"

#include <faiss/IndexFlat.h>
#include <faiss/IndexIDMap.h>

#include <QDataStream>
#include <QDir>
#include <QFile>
#include <QRegularExpression>
#include <QSqlError>
#include <QSqlQuery>
#include <QVariant>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace {

void normalizeL2(std::vector<float> &vector)
{
    double sum = 0.0;
    for (float value : vector)
        sum += static_cast<double>(value) * static_cast<double>(value);
    if (sum <= 0.0)
        return;
    const float scale = static_cast<float>(1.0 / std::sqrt(sum));
    for (float &value : vector)
        value *= scale;
}

QString sqlError(const QSqlQuery &query)
{
    return query.lastError().text();
}

constexpr quint32 kIndexMagic = 0x52414731;

} // namespace

KnowledgeBase::KnowledgeBase(QObject *parent)
    : QObject(parent)
{
}

KnowledgeBase::~KnowledgeBase()
{
    closeConnection();
}

QString KnowledgeBase::directory() const
{
    return m_directory;
}

QString KnowledgeBase::problem() const
{
    return m_problem;
}

bool KnowledgeBase::open(const QString &directory, QString *error)
{
    closeConnection();
    m_index.reset();
    m_problem.clear();

    QDir dir(directory);
    if (!dir.mkpath(QStringLiteral("."))) {
        if (error)
            *error = QStringLiteral("Не удалось создать каталог базы: %1").arg(directory);
        return false;
    }

    m_directory = dir.absolutePath();
    m_indexPath = dir.filePath(QStringLiteral("vectors.faiss"));

    if (!QSqlDatabase::isDriverAvailable(QStringLiteral("QSQLITE"))) {
        if (error)
            *error = QStringLiteral("Драйвер QSQLITE не найден. Проверьте плагин Qt sqldrivers.");
        return false;
    }

    m_db = QSqlDatabase::addDatabase(QStringLiteral("QSQLITE"), QStringLiteral("rag_knowledge"));
    m_db.setDatabaseName(dir.filePath(QStringLiteral("knowledge.db")));
    if (!m_db.open()) {
        if (error)
            *error = m_db.lastError().text();
        closeConnection();
        return false;
    }

    if (!createSchema(error)) {
        closeConnection();
        return false;
    }

    if (QFile::exists(m_indexPath)) {
        QString readError;
        if (!loadIndex(&readError)) {
            m_index.reset();
            m_problem = readError;
        }
    }

    refreshConsistency();
    return true;
}

bool KnowledgeBase::createSchema(QString *error)
{
    QSqlQuery query(m_db);
    const QStringList statements = {
        QStringLiteral("PRAGMA foreign_keys = ON"),
        QStringLiteral(
            "CREATE TABLE IF NOT EXISTS chunks ("
            "id INTEGER PRIMARY KEY,"
            "source_id INTEGER NOT NULL,"
            "question TEXT NOT NULL,"
            "context_text TEXT NOT NULL,"
            "answer TEXT NOT NULL,"
            "chunk_index INTEGER NOT NULL)"),
        QStringLiteral(
            "CREATE TABLE IF NOT EXISTS meta ("
            "key TEXT PRIMARY KEY,"
            "value TEXT NOT NULL)"),
        QStringLiteral(
            "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5("
            "context_text, question, content='chunks', content_rowid='id', tokenize='unicode61')"),
        QStringLiteral(
            "CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN "
            "INSERT INTO chunks_fts(rowid, context_text, question) "
            "VALUES (new.id, new.context_text, new.question); END"),
        QStringLiteral(
            "CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN "
            "INSERT INTO chunks_fts(chunks_fts, rowid, context_text, question) "
            "VALUES ('delete', old.id, old.context_text, old.question); END"),
    };

    for (const QString &statement : statements) {
        if (!query.exec(statement)) {
            if (error)
                *error = sqlError(query);
            return false;
        }
    }

    query.exec(QStringLiteral("SELECT COUNT(*) FROM chunks"));
    const int rows = query.next() ? query.value(0).toInt() : 0;
    query.exec(QStringLiteral("SELECT COUNT(*) FROM chunks_fts"));
    const int ftsRows = query.next() ? query.value(0).toInt() : 0;
    if (rows > 0 && ftsRows == 0) {
        if (!query.exec(QStringLiteral("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')"))) {
            if (error)
                *error = sqlError(query);
            return false;
        }
    }
    return true;
}

void KnowledgeBase::refreshConsistency()
{
    if (!m_problem.isEmpty())
        return;
    const int rows = chunkCount();
    const int vectors = m_index ? static_cast<int>(m_index->ntotal) : 0;
    if (rows != vectors) {
        m_problem = QStringLiteral(
                        "Индекс FAISS (%1 векторов) не совпадает с SQLite (%2 чанков). "
                        "Очистите базу и импортируйте файл заново.")
                        .arg(vectors)
                        .arg(rows);
    }
}

int KnowledgeBase::chunkCount()
{
    if (!m_db.isOpen())
        return 0;
    QSqlQuery query(m_db);
    if (!query.exec(QStringLiteral("SELECT COUNT(*) FROM chunks")) || !query.next())
        return 0;
    return query.value(0).toInt();
}

int KnowledgeBase::dimension() const
{
    return m_index ? static_cast<int>(m_index->d) : 0;
}

bool KnowledgeBase::ensureIndex(int dim, QString *error)
{
    if (dim <= 0) {
        if (error)
            *error = QStringLiteral("Пустой вектор эмбеддинга.");
        return false;
    }
    if (m_index) {
        if (static_cast<int>(m_index->d) != dim) {
            if (error) {
                *error = QStringLiteral("Размерность вектора %1 не совпадает с индексом %2.")
                             .arg(dim)
                             .arg(static_cast<int>(m_index->d));
            }
            return false;
        }
        return true;
    }

    try {
        auto *flat = new faiss::IndexFlatIP(dim);
        auto *mapped = new faiss::IndexIDMap(flat);
        mapped->own_fields = true;
        m_index.reset(mapped);
    } catch (const std::exception &ex) {
        if (error)
            *error = QString::fromUtf8(ex.what());
        return false;
    }
    return true;
}

qint64 KnowledgeBase::nextId()
{
    QSqlQuery query(m_db);
    if (query.exec(QStringLiteral("SELECT value FROM meta WHERE key = 'next_id'")) && query.next())
        return query.value(0).toLongLong();

    if (query.exec(QStringLiteral("SELECT COALESCE(MAX(id), 0) + 1 FROM chunks")) && query.next())
        return query.value(0).toLongLong();
    return 1;
}

bool KnowledgeBase::setMeta(const QString &key, const QString &value, QString *error)
{
    QSqlQuery query(m_db);
    query.prepare(QStringLiteral("INSERT INTO meta(key, value) VALUES(?, ?) "
                                 "ON CONFLICT(key) DO UPDATE SET value = excluded.value"));
    query.addBindValue(key);
    query.addBindValue(value);
    if (!query.exec()) {
        if (error)
            *error = sqlError(query);
        return false;
    }
    return true;
}

bool KnowledgeBase::addChunk(const ChunkRecord &chunk, const std::vector<float> &vector, QString *error)
{
    if (!m_problem.isEmpty()) {
        if (error)
            *error = m_problem;
        return false;
    }
    if (!ensureIndex(static_cast<int>(vector.size()), error))
        return false;

    std::vector<float> normalized = vector;
    normalizeL2(normalized);

    const qint64 id = nextId();
    if (!m_db.transaction()) {
        if (error)
            *error = m_db.lastError().text();
        return false;
    }

    QSqlQuery query(m_db);
    query.prepare(QStringLiteral(
        "INSERT INTO chunks(id, source_id, question, context_text, answer, chunk_index) "
        "VALUES(?, ?, ?, ?, ?, ?)"));
    query.addBindValue(id);
    query.addBindValue(chunk.sourceId);
    query.addBindValue(chunk.question);
    query.addBindValue(chunk.contextText);
    query.addBindValue(chunk.answer);
    query.addBindValue(chunk.chunkIndex);
    if (!query.exec()) {
        const QString message = sqlError(query);
        m_db.rollback();
        if (error)
            *error = message;
        return false;
    }

    try {
        const faiss::idx_t faissId = id;
        m_index->add_with_ids(1, normalized.data(), &faissId);
    } catch (const std::exception &ex) {
        m_db.rollback();
        if (error)
            *error = QString::fromUtf8(ex.what());
        return false;
    }

    if (!setMeta(QStringLiteral("next_id"), QString::number(id + 1), error)) {
        m_db.rollback();
        return false;
    }
    if (!m_db.commit()) {
        if (error)
            *error = m_db.lastError().text();
        return false;
    }
    return true;
}

bool KnowledgeBase::loadIndex(QString *error)
{
    QFile file(m_indexPath);
    if (!file.open(QIODevice::ReadOnly)) {
        if (error)
            *error = file.errorString();
        return false;
    }

    QDataStream in(&file);
    in.setVersion(QDataStream::Qt_6_5);
    quint32 magic = 0;
    qint32 dim = 0;
    qint64 count = 0;
    in >> magic >> dim >> count;
    if (in.status() != QDataStream::Ok || magic != kIndexMagic || dim <= 0 || count < 0) {
        if (error)
            *error = QStringLiteral("Файл индекса FAISS повреждён.");
        return false;
    }

    std::vector<faiss::idx_t> ids(static_cast<size_t>(count));
    for (qint64 i = 0; i < count; ++i) {
        qint64 id = 0;
        in >> id;
        if (in.status() != QDataStream::Ok) {
            if (error)
                *error = QStringLiteral("Файл индекса FAISS обрезан.");
            return false;
        }
        ids[static_cast<size_t>(i)] = id;
    }

    std::vector<float> vectors(static_cast<size_t>(count * dim));
    qint64 read = 0;
    const qint64 bytes = count * static_cast<qint64>(dim) * static_cast<qint64>(sizeof(float));
    auto *dest = reinterpret_cast<char *>(vectors.data());
    while (read < bytes) {
        const int chunk = static_cast<int>(std::min<qint64>(1 << 20, bytes - read));
        const int got = in.readRawData(dest + read, chunk);
        if (got != chunk) {
            if (error)
                *error = QStringLiteral("Файл индекса FAISS обрезан.");
            return false;
        }
        read += got;
    }

    if (!ensureIndex(dim, error))
        return false;
    if (count == 0)
        return true;

    try {
        m_index->add_with_ids(count, vectors.data(), ids.data());
    } catch (const std::exception &ex) {
        if (error)
            *error = QString::fromUtf8(ex.what());
        return false;
    }
    return true;
}

bool KnowledgeBase::saveIndex(QString *error)
{
    if (!m_index)
        return true;

    auto *mapped = dynamic_cast<faiss::IndexIDMap *>(m_index.get());
    auto *flat = mapped ? dynamic_cast<faiss::IndexFlat *>(mapped->index) : nullptr;
    if (!mapped || !flat || static_cast<qint64>(mapped->id_map.size()) != m_index->ntotal) {
        if (error)
            *error = QStringLiteral("Внутренний индекс FAISS не является IndexIDMap/IndexFlatIP.");
        return false;
    }

    QFile file(m_indexPath);
    if (!file.open(QIODevice::WriteOnly)) {
        if (error)
            *error = file.errorString();
        return false;
    }

    QDataStream out(&file);
    out.setVersion(QDataStream::Qt_6_5);
    out << kIndexMagic << qint32(m_index->d) << qint64(m_index->ntotal);
    for (faiss::idx_t id : mapped->id_map)
        out << qint64(id);

    const qint64 bytes = m_index->ntotal * static_cast<qint64>(m_index->d) * static_cast<qint64>(sizeof(float));
    qint64 written = 0;
    const auto *src = reinterpret_cast<const char *>(flat->get_xb());
    while (written < bytes) {
        const int chunk = static_cast<int>(std::min<qint64>(1 << 20, bytes - written));
        const int got = out.writeRawData(src + written, chunk);
        if (got != chunk) {
            if (error)
                *error = QStringLiteral("Не удалось записать индекс FAISS.");
            return false;
        }
        written += got;
    }
    return out.status() == QDataStream::Ok;
}

bool KnowledgeBase::clear(QString *error)
{
    if (!m_db.isOpen()) {
        if (error)
            *error = QStringLiteral("База не открыта.");
        return false;
    }
    QSqlQuery query(m_db);
    if (!query.exec(QStringLiteral("DELETE FROM chunks"))) {
        if (error)
            *error = sqlError(query);
        return false;
    }
    if (!setMeta(QStringLiteral("next_id"), QStringLiteral("1"), error))
        return false;

    m_index.reset();
    QFile::remove(m_indexPath);
    m_problem.clear();
    return true;
}

ChunkRecord KnowledgeBase::loadChunk(qint64 id)
{
    ChunkRecord chunk;
    if (!m_db.isOpen())
        return chunk;

    QSqlQuery query(m_db);
    query.prepare(QStringLiteral(
        "SELECT id, source_id, question, context_text, answer, chunk_index "
        "FROM chunks WHERE id = ?"));
    query.addBindValue(id);
    if (!query.exec() || !query.next())
        return chunk;

    chunk.id = query.value(0).toLongLong();
    chunk.sourceId = query.value(1).toLongLong();
    chunk.question = query.value(2).toString();
    chunk.contextText = query.value(3).toString();
    chunk.answer = query.value(4).toString();
    chunk.chunkIndex = query.value(5).toInt();
    return chunk;
}

QVector<ScoredChunk> KnowledgeBase::searchByVector(const std::vector<float> &vector, int k, QString *error)
{
    if (!m_problem.isEmpty()) {
        if (error)
            *error = m_problem;
        return {};
    }
    if (!m_index || m_index->ntotal <= 0 || k <= 0)
        return {};
    if (static_cast<int>(vector.size()) != static_cast<int>(m_index->d)) {
        if (error) {
            *error = QStringLiteral("Размерность запроса %1 не совпадает с индексом %2.")
                         .arg(vector.size())
                         .arg(static_cast<int>(m_index->d));
        }
        return {};
    }

    std::vector<float> normalized = vector;
    normalizeL2(normalized);
    const int limit = std::min(k, static_cast<int>(m_index->ntotal));
    std::vector<float> distances(static_cast<size_t>(limit));
    std::vector<faiss::idx_t> labels(static_cast<size_t>(limit), -1);

    try {
        m_index->search(1, normalized.data(), limit, distances.data(), labels.data());
    } catch (const std::exception &ex) {
        if (error)
            *error = QString::fromUtf8(ex.what());
        return {};
    }

    QVector<ScoredChunk> hits;
    for (int i = 0; i < limit; ++i) {
        if (labels[static_cast<size_t>(i)] < 0)
            continue;
        ScoredChunk hit;
        hit.chunk = loadChunk(labels[static_cast<size_t>(i)]);
        hit.score = distances[static_cast<size_t>(i)];
        hit.via = QStringLiteral("vector");
        if (hit.chunk.id >= 0)
            hits.append(hit);
    }
    return hits;
}

QString KnowledgeBase::buildFtsQuery(const QString &text)
{
    const QRegularExpression word(QStringLiteral("(?:\\p{L}|\\p{N}){2,}"),
                                  QRegularExpression::UseUnicodePropertiesOption);
    QStringList terms;
    auto it = word.globalMatch(text);
    while (it.hasNext()) {
        QString token = it.next().captured().toLower();
        const bool latin = std::any_of(token.cbegin(), token.cend(), [](QChar ch) {
            return ch.script() == QChar::Script_Latin;
        });
        if (token.size() < 3 && !latin)
            continue;
        token.replace(u'"', QString());
        if (!token.isEmpty())
            terms.append(u'"' + token + u'"');
        if (terms.size() >= 12)
            break;
    }
    terms.removeDuplicates();
    return terms.join(QStringLiteral(" OR "));
}

QVector<ScoredChunk> KnowledgeBase::searchByText(const QString &query, int k, QString *error)
{
    const QString match = buildFtsQuery(query);
    if (match.isEmpty() || k <= 0)
        return {};

    QSqlQuery sql(m_db);
    sql.prepare(QStringLiteral(
        "SELECT chunks.id, bm25(chunks_fts) AS rank "
        "FROM chunks_fts "
        "JOIN chunks ON chunks.id = chunks_fts.rowid "
        "WHERE chunks_fts MATCH ? "
        "ORDER BY rank "
        "LIMIT ?"));
    sql.addBindValue(match);
    sql.addBindValue(k);
    if (!sql.exec()) {
        if (error)
            *error = sqlError(sql);
        return {};
    }

    QVector<ScoredChunk> hits;
    while (sql.next()) {
        ScoredChunk hit;
        hit.chunk = loadChunk(sql.value(0).toLongLong());
        hit.score = static_cast<float>(sql.value(1).toDouble());
        hit.via = QStringLiteral("fts");
        if (hit.chunk.id >= 0)
            hits.append(hit);
    }
    return hits;
}

void KnowledgeBase::closeConnection()
{
    const QString name = m_db.connectionName();
    if (m_db.isOpen())
        m_db.close();
    m_db = QSqlDatabase();
    if (!name.isEmpty() && QSqlDatabase::contains(name))
        QSqlDatabase::removeDatabase(name);
}
