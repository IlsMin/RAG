#include "RagPipeline.h"

#include "KnowledgeBase.h"
#include "LlamaClient.h"
#include "TextChunker.h"

#include <QFile>
#include <QHash>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QRegularExpression>
#include <QSet>

#include <algorithm>

namespace {

constexpr int kSaveEvery = 20;
constexpr int kRerankCandidates = 6;
constexpr int kExtraQueries = 2;
constexpr int kMultiQueryTries = 3;

bool hasForeignScript(const QString &text)
{
    for (const QChar ch : text) {
        const char16_t code = ch.unicode();
        const bool cjk = (code >= 0x3400 && code <= 0x9FFF) || (code >= 0xF900 && code <= 0xFAFF)
            || (code >= 0x3000 && code <= 0x303F) || (code >= 0x3040 && code <= 0x30FF)
            || (code >= 0xAC00 && code <= 0xD7AF) || (code >= 0xFF00 && code <= 0xFFEF);
        if (cjk)
            return true;
    }
    return false;
}

int cyrillicLetters(const QString &text)
{
    int count = 0;
    for (const QChar ch : text) {
        const char16_t code = ch.unicode();
        if (code >= 0x0400 && code <= 0x04FF)
            ++count;
    }
    return count;
}

QString firstLine(const QString &text, int limit)
{
    QString line = text.section(u'\n', 0, 0).simplified();
    if (line.size() > limit)
        line = line.left(limit).trimmed() + QStringLiteral("…");
    return line;
}

QString formatHits(const QVector<ScoredChunk> &hits, bool higherIsBetter)
{
    QString body;
    for (int i = 0; i < hits.size(); ++i) {
        const ScoredChunk &hit = hits.at(i);
        body += QStringLiteral("%1. id=%2  оценка=%3  %4\n")
                    .arg(i + 1)
                    .arg(hit.chunk.id)
                    .arg(QString::number(hit.score, 'f', 3))
                    .arg(firstLine(hit.chunk.question, 90));
    }
    if (body.isEmpty())
        body = QStringLiteral("Ничего не найдено.");
    else if (!higherIsBetter)
        body += QStringLiteral("\nДля полнотекста меньше bm25 — ближе к запросу.");
    return body.trimmed();
}

QVector<ScoredChunk> fuseRrf(const QVector<QVector<ScoredChunk>> &rankings, KnowledgeBase *knowledge)
{
    constexpr float kRrf = 60.f;
    QHash<qint64, float> scores;
    QHash<qint64, ChunkRecord> records;
    for (const QVector<ScoredChunk> &ranking : rankings) {
        for (int i = 0; i < ranking.size(); ++i) {
            const ScoredChunk &hit = ranking.at(i);
            scores[hit.chunk.id] += 1.f / (kRrf + static_cast<float>(i + 1));
            records.insert(hit.chunk.id, hit.chunk);
        }
    }

    QVector<ScoredChunk> fused;
    fused.reserve(scores.size());
    for (auto it = scores.cbegin(); it != scores.cend(); ++it) {
        ScoredChunk hit;
        hit.chunk = records.value(it.key());
        if (hit.chunk.id < 0)
            hit.chunk = knowledge->loadChunk(it.key());
        hit.score = it.value();
        hit.via = QStringLiteral("rrf");
        if (hit.chunk.id >= 0)
            fused.append(hit);
    }
    std::sort(fused.begin(), fused.end(), [](const ScoredChunk &a, const ScoredChunk &b) {
        return a.score > b.score;
    });
    return fused;
}

int answerSubstance(const QString &contextText)
{
    const int split = contextText.indexOf(u'\n');
    const QString body = (split < 0 ? contextText : contextText.mid(split + 1)).trimmed();
    return body.size();
}

QVector<ScoredChunk> strongFtsHits(const QVector<ScoredChunk> &hits)
{
    if (hits.size() < 2)
        return {};

    const float best = hits.first().score;
    QVector<ScoredChunk> strong;
    for (const ScoredChunk &hit : hits) {
        if (hit.score > best * 0.75f)
            break;
        strong.append(hit);
    }
    if (strong.size() >= hits.size())
        return {};

    std::sort(strong.begin(), strong.end(), [](const ScoredChunk &left, const ScoredChunk &right) {
        return answerSubstance(left.chunk.contextText) > answerSubstance(right.chunk.contextText);
    });
    return strong;
}

QString answerPrompt(const QString &question, const QVector<ChunkRecord> &chunks)
{
    QString context;
    for (int i = 0; i < chunks.size(); ++i) {
        if (i > 0)
            context += QStringLiteral("\n\n");
        context += chunks.at(i).contextText;
    }

    return QStringLiteral(
               "Ты — строгое профессиональное ИИ-руководство для IT-специалистов. "
               "Твоя задача — отвечать на вопросы пользователя, используя ТОЛЬКО предоставленный ниже контекст.\n"
               "\n"
               "ПРАВИЛА ОТВЕТА:\n"
               "1. Отвечай кратко, технически точно и по делу.\n"
               "2. Используй профессиональный IT-сленг из контекста, если это необходимо.\n"
               "3. Если в контексте нет прямого ответа на вопрос или информации недостаточно, ответь строго фразой: "
               "\"Извините, в моей базе знаний нет информации для ответа на этот вопрос.\"\n"
               "4. Ничего не придумывай от себя и не используй свои внешние знания.\n"
               "\n"
               "КОНТЕКСТ ДЛЯ ОТВЕТА:\n"
               "---\n"
               "%1\n"
               "---\n"
               "\n"
               "ВОПРОС ПОЛЬЗОВАТЕЛЯ:\n"
               "%2\n"
               "\n"
               "ОТВЕТ:")
        .arg(context, question);
}

} // namespace

RagPipeline::RagPipeline(KnowledgeBase *knowledge, LlamaClient *client, QObject *parent)
    : QObject(parent)
    , m_knowledge(knowledge)
    , m_client(client)
{
    connect(m_client, &LlamaClient::embeddingReady, this, &RagPipeline::onEmbedding);
    connect(m_client, &LlamaClient::chatReady, this, &RagPipeline::onChat);
    connect(m_client, &LlamaClient::requestFailed, this, &RagPipeline::onFailed);
}

void RagPipeline::ask(const QString &question, const RagSettings &settings)
{
    if (m_phase != Phase::Idle) {
        emit errorOccurred(QStringLiteral("Дождитесь окончания текущей операции."));
        return;
    }
    const QString trimmed = question.trimmed();
    if (trimmed.isEmpty()) {
        emit errorOccurred(QStringLiteral("Введите вопрос."));
        return;
    }
    if (!settings.direct && !m_knowledge->problem().isEmpty()) {
        emit errorOccurred(m_knowledge->problem());
        return;
    }
    if (!settings.direct && m_knowledge->chunkCount() <= 0) {
        emit errorOccurred(QStringLiteral("База пуста. Импортируйте questions.jsonl на вкладке «Настройки»."));
        return;
    }

    m_settings = settings;
    m_question = trimmed;
    m_cancelled = false;
    m_client->setChatBaseUrl(settings.chatBaseUrl);
    m_client->setEmbeddingBaseUrl(settings.embeddingBaseUrl);

    if (settings.direct) {
        emit stepReady(QStringLiteral("Запрос"),
                       trimmed + QStringLiteral("\n\nрежим: прямой запрос к модели, без FAISS и без контекста базы"));
        emit busyChanged(true);
        emit stepReady(QStringLiteral("Промпт генерации"), trimmed);
        m_phase = Phase::Answer;
        m_activeRequest = m_client->requestChat(
            trimmed, 400, 0.2, QStringLiteral("Отвечай на русском языке."));
        return;
    }

    QStringList techniques;
    techniques << (settings.multiQuery ? QStringLiteral("multi-query: включён")
                                       : QStringLiteral("multi-query: выключен"));
    techniques << (settings.hybridSearch ? QStringLiteral("hybrid search: включён")
                                         : QStringLiteral("hybrid search: выключен"));
    techniques << (settings.rerank ? QStringLiteral("rerank: включён") : QStringLiteral("rerank: выключен"));
    emit stepReady(QStringLiteral("Запрос"),
                   trimmed + QStringLiteral("\n\n") + techniques.join(u'\n'));
    emit busyChanged(true);

    if (settings.multiQuery) {
        m_multiQueryAttempt = 0;
        requestMultiQuery();
        return;
    }
    beginQueryEmbeddings({trimmed});
}

void RagPipeline::requestMultiQuery()
{
    m_phase = Phase::MultiQuery;
    const QString prompt = QStringLiteral(
                               "Переформулируй вопрос ровно двумя разными короткими поисковыми запросами. "
                               "Пиши только по-русски, кириллицей. Китайский, английский и другие языки запрещены. "
                               "Латиница допустима только внутри имён вроде QNX. "
                               "Не повторяй исходную формулировку. "
                               "Ответь ровно двумя строками, без нумерации и без пояснений.\n\nВопрос: %1")
                               .arg(m_question);
    const QString system = QStringLiteral(
        "Ты переписываешь поисковые запросы только на русском языке. "
        "Никогда не используй китайские иероглифы.");
    const double temperature = m_multiQueryAttempt == 0 ? 0.2 : 0.7;
    m_activeRequest = m_client->requestChat(prompt, 160, temperature, system);
}

void RagPipeline::importJsonl(const QString &path, const RagSettings &settings)
{
    if (m_phase != Phase::Idle) {
        emit errorOccurred(QStringLiteral("Дождитесь окончания текущей операции."));
        return;
    }
    if (!m_knowledge->problem().isEmpty()) {
        emit errorOccurred(m_knowledge->problem());
        return;
    }

    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        emit errorOccurred(QStringLiteral("Не удалось открыть файл: %1").arg(file.errorString()));
        return;
    }

    m_settings = settings;
    m_cancelled = false;
    m_client->setEmbeddingBaseUrl(settings.embeddingBaseUrl);
    m_importQueue.clear();

    int accepted = 0;
    int skipped = 0;
    int bad = 0;
    const int limit = std::max(1, settings.importLimit);
    while (!file.atEnd() && accepted < limit) {
        QByteArray line = file.readLine();
        if (line.trimmed().isEmpty())
            continue;
        if (line.size() > 5000000) {
            ++bad;
            continue;
        }
        if (line.startsWith("\xEF\xBB\xBF"))
            line.remove(0, 3);

        QJsonParseError parseError;
        const QJsonDocument document = QJsonDocument::fromJson(line, &parseError);
        if (parseError.error != QJsonParseError::NoError || !document.isObject()) {
            ++bad;
            continue;
        }
        const QVector<PendingChunk> chunks = chunksFromRecord(document.object());
        if (chunks.isEmpty()) {
            ++skipped;
            continue;
        }
        ++accepted;
        for (const PendingChunk &chunk : chunks)
            m_importQueue.enqueue(chunk);
    }

    if (m_importQueue.isEmpty()) {
        emit errorOccurred(QStringLiteral("В файле не нашлось вопросов с непустым текстом ответа."));
        return;
    }

    m_importDone = 0;
    m_importTotal = m_importQueue.size();
    m_phase = Phase::ImportEmbedding;
    emit busyChanged(true);
    emit stepReady(QStringLiteral("Импорт"),
                   QStringLiteral("Файл: %1\nВопросов с ответами: %2\nЧанков в очереди: %3\n"
                                  "Пропущено без ответа: %4\nБитых строк: %5")
                       .arg(path)
                       .arg(accepted)
                       .arg(m_importTotal)
                       .arg(skipped)
                       .arg(bad));
    emit importProgress(0, m_importTotal);
    embedNextImportChunk();
}

void RagPipeline::cancel()
{
    if (m_phase == Phase::Idle)
        return;
    m_cancelled = true;
    m_client->abort();
}

QVector<RagPipeline::PendingChunk> RagPipeline::chunksFromRecord(const QJsonObject &record) const
{
    const QJsonArray answers = record.value(QStringLiteral("answers")).toArray();
    if (answers.isEmpty())
        return {};

    const QString title = record.value(QStringLiteral("title")).toString().trimmed();
    const QString description = TextChunker::htmlToPlain(record.value(QStringLiteral("description")).toString());
    QString question = title;
    if (!description.isEmpty()) {
        if (!question.isEmpty())
            question += QStringLiteral("\n\n");
        question += description;
    }
    if (question.isEmpty())
        question = QStringLiteral("(без заголовка)");

    QString prefixTitle = title.isEmpty() ? firstLine(question, 180) : title;
    if (prefixTitle.size() > 180)
        prefixTitle = prefixTitle.left(180).trimmed() + QStringLiteral("…");

    QVector<QJsonObject> ordered;
    QVector<QJsonObject> rest;
    for (const QJsonValue &value : answers) {
        if (!value.isObject())
            continue;
        const QJsonObject answer = value.toObject();
        if (answer.value(QStringLiteral("accepted")).toBool())
            ordered.append(answer);
        else
            rest.append(answer);
    }
    ordered += rest;

    QVector<PendingChunk> chunks;
    int answerNo = 0;
    for (const QJsonObject &answer : ordered) {
        const QString body = TextChunker::htmlToPlain(answer.value(QStringLiteral("body")).toString());
        if (body.isEmpty())
            continue;
        ++answerNo;
        const QStringList parts = TextChunker::split(body);
        for (int i = 0; i < parts.size(); ++i) {
            PendingChunk chunk;
            chunk.sourceId = record.value(QStringLiteral("id")).toVariant().toLongLong();
            chunk.question = question;
            chunk.answer = body;
            chunk.chunkIndex = i;
            chunk.contextText = QStringLiteral("[Вопрос с Хабра: %1 | Раздел ответа: ответ %2, фрагмент %3/%4]\n%5")
                                    .arg(prefixTitle)
                                    .arg(answerNo)
                                    .arg(i + 1)
                                    .arg(parts.size())
                                    .arg(parts.at(i));
            chunks.append(chunk);
        }
    }
    return chunks;
}

void RagPipeline::embedNextImportChunk()
{
    if (m_cancelled || m_importQueue.isEmpty()) {
        finishImport();
        return;
    }
    m_phase = Phase::ImportEmbedding;
    m_activeRequest = m_client->requestEmbedding(m_importQueue.head().contextText);
}

void RagPipeline::finishImport()
{
    const bool cancelled = m_cancelled;
    QString saveError;
    if (!m_knowledge->saveIndex(&saveError) && !saveError.isEmpty())
        emit errorOccurred(QStringLiteral("Чанки записаны, но индекс FAISS не сохранился: %1").arg(saveError));

    const int written = m_importDone;
    m_importQueue.clear();
    m_cancelled = false;
    m_activeRequest = 0;
    m_phase = Phase::Idle;
    emit importProgress(written, std::max(written, m_importTotal));
    emit importFinished(written);
    emit stepReady(cancelled ? QStringLiteral("Импорт остановлен") : QStringLiteral("Импорт завершён"),
                   QStringLiteral("Записано чанков: %1").arg(written));
    emit busyChanged(false);
}

void RagPipeline::beginQueryEmbeddings(const QStringList &queries)
{
    m_queries = queries;
    m_queryVectors.clear();
    m_queryVectors.reserve(queries.size());
    m_embedPos = 0;
    m_phase = Phase::QueryEmbedding;
    embedNextQuery();
}

void RagPipeline::embedNextQuery()
{
    if (m_embedPos >= m_queries.size()) {
        runSearch();
        return;
    }
    m_phase = Phase::QueryEmbedding;
    m_activeRequest = m_client->requestEmbedding(m_queries.at(m_embedPos));
}

void RagPipeline::runSearch()
{
    QVector<QVector<ScoredChunk>> rankings;
    QString searchLog;
    const int k = std::max(1, m_settings.retrieveK);

    for (int i = 0; i < m_queries.size(); ++i) {
        QString error;
        const QVector<ScoredChunk> hits = m_knowledge->searchByVector(m_queryVectors.at(i), k, &error);
        rankings.append(hits);
        searchLog += QStringLiteral("Запрос %1: %2\n").arg(i + 1).arg(m_queries.at(i));
        if (!error.isEmpty())
            searchLog += QStringLiteral("Ошибка: %1\n").arg(error);
        searchLog += formatHits(hits, true) + QStringLiteral("\n\n");
    }
    emit stepReady(QStringLiteral("Векторный поиск (FAISS, inner product)"), searchLog.trimmed());

    QVector<ScoredChunk> ftsHits;
    if (m_settings.hybridSearch) {
        QString error;
        const QString fts = KnowledgeBase::buildFtsQuery(m_question);
        ftsHits = m_knowledge->searchByText(m_question, k, &error);
        const QVector<ScoredChunk> hits = ftsHits;
        rankings.append(hits);
        QString body = QStringLiteral("FTS-запрос: %1\n").arg(fts.isEmpty() ? QStringLiteral("(пусто)") : fts);
        if (!error.isEmpty())
            body += QStringLiteral("Ошибка FTS: %1\n").arg(error);
        body += formatHits(hits, false);
        emit stepReady(QStringLiteral("Полнотекст (SQLite FTS5)"), body.trimmed());
    }

    m_fused = fuseRrf(rankings, m_knowledge);
    m_pinned.clear();
    const QVector<ScoredChunk> pinned = strongFtsHits(ftsHits);
    if (!pinned.isEmpty()) {
        QSet<qint64> seen;
        QVector<ScoredChunk> merged;
        QStringList pinnedIds;
        for (const ScoredChunk &hit : pinned) {
            if (seen.contains(hit.chunk.id))
                continue;
            seen.insert(hit.chunk.id);
            merged.append(hit);
            m_pinned.append(hit.chunk);
            pinnedIds << QString::number(hit.chunk.id);
        }
        for (const ScoredChunk &hit : m_fused) {
            if (seen.contains(hit.chunk.id))
                continue;
            seen.insert(hit.chunk.id);
            merged.append(hit);
        }
        m_fused = merged;
        emit stepReady(QStringLiteral("Приоритет полнотекста"),
                       QStringLiteral("У первых результатов FTS большой отрыв. "
                                      "В контекст сначала пойдут более содержательные из них: %1")
                           .arg(pinnedIds.join(QStringLiteral(", "))));
    }
    if (m_fused.size() > kRerankCandidates)
        m_fused.resize(kRerankCandidates);

    emit stepReady(QStringLiteral("Слияние списков (Reciprocal Rank Fusion)"), formatHits(m_fused, true));

    if (m_fused.isEmpty()) {
        fail(QStringLiteral("Поиск не вернул ни одного чанка."));
        return;
    }
    if (m_settings.rerank)
        beginRerank();
    else {
        QVector<ChunkRecord> chosen;
        const int take = std::min(std::max(1, m_settings.contextChunks), static_cast<int>(m_fused.size()));
        for (int i = 0; i < take; ++i)
            chosen.append(m_fused.at(i).chunk);
        beginAnswer(withPinned(chosen));
    }
}

void RagPipeline::beginRerank()
{
    const int take = std::min(kRerankCandidates, static_cast<int>(m_fused.size()));
    QString listing;
    for (int i = 0; i < take; ++i) {
        const ChunkRecord &chunk = m_fused.at(i).chunk;
        QString excerpt = chunk.contextText;
        excerpt.replace(u'\n', u' ');
        if (excerpt.size() > 280)
            excerpt = excerpt.left(280).trimmed() + QStringLiteral("…");
        listing += QStringLiteral("id=%1: %2\n").arg(chunk.id).arg(excerpt);
    }

    const QString prompt = QStringLiteral(
                               "Выбери фрагменты, которые лучше всего помогают ответить на вопрос. "
                               "Верни номера двух лучших, по одному на строку, строго в формате id=ЧИСЛО. "
                               "Ничего больше не пиши.\n\nВопрос: %1\n\nФрагменты:\n%2")
                               .arg(m_question, listing);
    m_phase = Phase::Rerank;
    m_activeRequest = m_client->requestChat(prompt, 64, 0.1);
}

void RagPipeline::beginAnswer(const QVector<ChunkRecord> &chunks)
{
    QString shown;
    for (const ChunkRecord &chunk : chunks) {
        shown += QStringLiteral("id=%1 source=%2\n%3\n\n").arg(chunk.id).arg(chunk.sourceId).arg(chunk.contextText);
    }
    const QString prompt = answerPrompt(m_question, chunks);
    emit stepReady(QStringLiteral("Фрагменты, отданные в Llama"), shown.trimmed());
    emit stepReady(QStringLiteral("Промпт генерации"), prompt);
    m_phase = Phase::Answer;
    m_activeRequest = m_client->requestChat(prompt, 220, 0.2);
}

void RagPipeline::onEmbedding(quint64 requestId, const QVector<float> &vector)
{
    if (requestId != m_activeRequest || m_cancelled)
        return;

    if (m_phase == Phase::ImportEmbedding) {
        if (m_importQueue.isEmpty())
            return;
        const PendingChunk pending = m_importQueue.dequeue();
        ChunkRecord record;
        record.sourceId = pending.sourceId;
        record.question = pending.question;
        record.contextText = pending.contextText;
        record.answer = pending.answer;
        record.chunkIndex = pending.chunkIndex;

        QString error;
        std::vector<float> stdVector(vector.cbegin(), vector.cend());
        if (!m_knowledge->addChunk(record, stdVector, &error)) {
            fail(error.isEmpty() ? QStringLiteral("Не удалось записать чанк.") : error);
            return;
        }
        ++m_importDone;
        emit importProgress(m_importDone, m_importTotal);
        if (m_importDone % kSaveEvery == 0) {
            QString saveError;
            if (!m_knowledge->saveIndex(&saveError)) {
                fail(QStringLiteral("Ошибка записи индекса FAISS: %1").arg(saveError));
                return;
            }
        }
        embedNextImportChunk();
        return;
    }

    if (m_phase == Phase::QueryEmbedding) {
        m_queryVectors.append(std::vector<float>(vector.cbegin(), vector.cend()));
        ++m_embedPos;
        emit stepReady(QStringLiteral("Эмбеддинг запроса %1/%2").arg(m_embedPos).arg(m_queries.size()),
                       QStringLiteral("«%1»\nРазмерность: %2").arg(m_queries.at(m_embedPos - 1)).arg(vector.size()));
        embedNextQuery();
    }
}

void RagPipeline::onChat(quint64 requestId, const QString &text)
{
    if (requestId != m_activeRequest || m_cancelled)
        return;

    if (m_phase == Phase::MultiQuery) {
        QStringList extras;
        for (const QString &variant : parseVariants(text, 8)) {
            if (variant.compare(m_question, Qt::CaseInsensitive) == 0)
                continue;
            bool duplicate = false;
            for (const QString &have : extras) {
                if (have.compare(variant, Qt::CaseInsensitive) == 0) {
                    duplicate = true;
                    break;
                }
            }
            if (duplicate)
                continue;
            extras.append(variant);
            if (extras.size() == kExtraQueries)
                break;
        }
        if (extras.size() < kExtraQueries && m_multiQueryAttempt + 1 < kMultiQueryTries) {
            ++m_multiQueryAttempt;
            emit stepReady(QStringLiteral("Multi-query"),
                           QStringLiteral("Модель вернула %1 из %2 дополнительных запросов. Повторяю.\n\n%3")
                               .arg(extras.size())
                               .arg(kExtraQueries)
                               .arg(text.trimmed()));
            requestMultiQuery();
            return;
        }
        QStringList queries;
        queries << m_question;
        queries += extras;
        emit stepReady(QStringLiteral("Multi-query"),
                       QStringLiteral("Ответ модели:\n%1\n\nДополнительных запросов: %2\nИщем по:\n- %3")
                           .arg(text.trimmed())
                           .arg(extras.size())
                           .arg(queries.join(QStringLiteral("\n- "))));
        beginQueryEmbeddings(queries);
        return;
    }

    if (m_phase == Phase::Rerank) {
        const QVector<qint64> ids = parseRankedIds(text);
        QSet<qint64> allowed;
        for (const ScoredChunk &hit : m_fused)
            allowed.insert(hit.chunk.id);

        QVector<ChunkRecord> chosen;
        for (qint64 id : ids) {
            if (!allowed.contains(id))
                continue;
            for (const ScoredChunk &hit : m_fused) {
                if (hit.chunk.id == id) {
                    chosen.append(hit.chunk);
                    break;
                }
            }
            if (chosen.size() >= std::max(1, m_settings.contextChunks))
                break;
        }
        if (chosen.isEmpty()) {
            const int take = std::min(std::max(1, m_settings.contextChunks), static_cast<int>(m_fused.size()));
            for (int i = 0; i < take; ++i)
                chosen.append(m_fused.at(i).chunk);
        }
        emit stepReady(QStringLiteral("Rerank"),
                       QStringLiteral("Ответ модели:\n%1\n\nВ контекст взяты id: %2")
                           .arg(text)
                           .arg([&chosen]() {
                               QStringList list;
                               for (const ChunkRecord &chunk : chosen)
                                   list << QString::number(chunk.id);
                               return list.join(QStringLiteral(", "));
                           }()));
        beginAnswer(withPinned(chosen));
        return;
    }

    if (m_phase == Phase::Answer) {
        emit stepReady(QStringLiteral("Ответ модели"), text);
        emit answerReady(text);
        becomeIdle();
    }
}

void RagPipeline::onFailed(quint64 requestId, const QString &error)
{
    if (m_cancelled) {
        if (m_phase == Phase::ImportEmbedding)
            finishImport();
        else {
            m_cancelled = false;
            m_activeRequest = 0;
            m_phase = Phase::Idle;
            emit stepReady(QStringLiteral("Остановлено"), QStringLiteral("Операция прервана."));
            emit busyChanged(false);
        }
        return;
    }
    if (requestId != m_activeRequest)
        return;

    if (m_phase == Phase::MultiQuery) {
        emit stepReady(QStringLiteral("Multi-query"),
                       QStringLiteral("Переформулировка не удалась (%1). Ищем по исходному вопросу.").arg(error));
        beginQueryEmbeddings({m_question});
        return;
    }
    if (m_phase == Phase::Rerank) {
        emit stepReady(QStringLiteral("Rerank"),
                       QStringLiteral("Переранжирование не удалось (%1). Берём порядок слияния.").arg(error));
        QVector<ChunkRecord> chosen;
        const int take = std::min(std::max(1, m_settings.contextChunks), static_cast<int>(m_fused.size()));
        for (int i = 0; i < take; ++i)
            chosen.append(m_fused.at(i).chunk);
        beginAnswer(withPinned(chosen));
        return;
    }
    fail(error);
}

QStringList RagPipeline::parseVariants(const QString &text, int limit)
{
    QStringList variants;
    const QRegularExpression lead(QStringLiteral("^(?:\\d+[\\.)]|[-*•])\\s*"));
    const QStringList lines = text.split(u'\n');
    for (QString line : lines) {
        line = line.trimmed();
        line.remove(lead);
        line = line.trimmed();
        if (line.size() < 8)
            continue;
        if (line.contains(u'{') || line.contains(u'}') || line.contains(QStringLiteral("//"))
            || line.contains(QStringLiteral("</")) || line.contains(QLatin1String("<MID"), Qt::CaseInsensitive)
            || line.contains(QLatin1String("<PRE"), Qt::CaseInsensitive))
            continue;
        int letters = 0;
        for (const QChar ch : line) {
            if (ch.isLetter())
                ++letters;
        }
        if (letters < 8)
            continue;
        if (hasForeignScript(line) || cyrillicLetters(line) < 6)
            continue;
        variants.append(line);
        if (variants.size() == limit)
            break;
    }
    return variants;
}

QVector<ChunkRecord> RagPipeline::withPinned(const QVector<ChunkRecord> &chosen) const
{
    QVector<ChunkRecord> result;
    QSet<qint64> seen;
    const int limit = std::max(1, m_settings.contextChunks);
    for (const ChunkRecord &chunk : m_pinned) {
        if (result.size() >= limit)
            break;
        if (seen.contains(chunk.id))
            continue;
        seen.insert(chunk.id);
        result.append(chunk);
    }
    for (const ChunkRecord &chunk : chosen) {
        if (result.size() >= limit)
            break;
        if (seen.contains(chunk.id))
            continue;
        seen.insert(chunk.id);
        result.append(chunk);
    }
    return result;
}

QVector<qint64> RagPipeline::parseRankedIds(const QString &text)
{
    QVector<qint64> ids;
    const QRegularExpression idRe(QStringLiteral("id\\s*[=:]\\s*(\\d+)"),
                                  QRegularExpression::CaseInsensitiveOption);
    auto it = idRe.globalMatch(text);
    while (it.hasNext()) {
        const qint64 id = it.next().captured(1).toLongLong();
        if (!ids.contains(id))
            ids.append(id);
    }
    if (!ids.isEmpty())
        return ids;

    const QRegularExpression lone(QStringLiteral("\\b(\\d+)\\b"));
    auto numbers = lone.globalMatch(text);
    while (numbers.hasNext()) {
        const qint64 id = numbers.next().captured(1).toLongLong();
        if (!ids.contains(id))
            ids.append(id);
    }
    return ids;
}

void RagPipeline::fail(const QString &message)
{
    m_importQueue.clear();
    m_cancelled = false;
    m_activeRequest = 0;
    m_phase = Phase::Idle;
    emit errorOccurred(message);
    emit busyChanged(false);
}

void RagPipeline::becomeIdle()
{
    m_cancelled = false;
    m_activeRequest = 0;
    m_phase = Phase::Idle;
    emit busyChanged(false);
}
