#include "mainwindow.h"

#include "KnowledgeBase.h"
#include "LlamaClient.h"
#include "TextChunker.h"

#include <QApplication>
#include <QCoreApplication>
#include <QDir>
#include <QEventLoop>
#include <QTemporaryDir>
#include <QTextStream>
#include <QTimer>
#include <QVector>

#include <vector>

namespace {

int fail(QTextStream &out, const QString &message)
{
    out << "SELF-TEST FAIL: " << message << "\n";
    return 1;
}

int runSelfTest()
{
    QTextStream out(stdout);
    const QString shortText = QStringLiteral("Короткий ответ про Qt.");
    const QStringList one = TextChunker::split(shortText);
    if (one.size() != 1 || one.first() != shortText)
        return fail(out, QStringLiteral("короткий текст должен остаться одним чанком"));

    QString longText;
    for (int i = 0; i < 40; ++i) {
        longText += QStringLiteral("Абзац %1 про сигналы и слоты Qt. SignalSlotConnect остаётся целиком. "
                                   "Дальше идёт пояснение, почему контекст на стыке нельзя терять.\n\n")
                        .arg(i + 1);
    }
    const QStringList chunks = TextChunker::split(longText);
    if (chunks.size() < 2)
        return fail(out, QStringLiteral("длинный текст не нарезался"));
    bool sawToken = false;
    for (const QString &chunk : chunks) {
        if (chunk.trimmed().isEmpty())
            return fail(out, QStringLiteral("пустой чанк"));
        if (chunk.size() > TextChunker::kChunkSize + TextChunker::kChunkOverlap)
            return fail(out, QStringLiteral("чанк длиннее %1 символов").arg(chunk.size()));
        if (chunk.contains(QStringLiteral("SignalSlotConnect")))
            sawToken = true;
        if (chunk.contains(QStringLiteral("SignalSlotCo")) && !chunk.contains(QStringLiteral("SignalSlotConnect")))
            return fail(out, QStringLiteral("слово разрезано"));
    }
    if (!sawToken)
        return fail(out, QStringLiteral("в чанках нет контрольного слова"));
    out << "chunker ok, chunks=" << chunks.size() << "\n";

    QTemporaryDir tempDir;
    if (!tempDir.isValid())
        return fail(out, QStringLiteral("не создан временный каталог"));

    KnowledgeBase knowledge;
    QString error;
    if (!knowledge.open(tempDir.path(), &error))
        return fail(out, error);

    LlamaClient client;
    client.setChatBaseUrl(QStringLiteral("http://127.0.0.1:8080"));
    client.setEmbeddingBaseUrl(QStringLiteral("http://127.0.0.1:8081"));

    QEventLoop loop;
    QVector<float> vector;
    QString netError;
    QObject::connect(&client, &LlamaClient::embeddingReady, &loop, [&](quint64, const QVector<float> &values) {
        vector = values;
        loop.quit();
    });
    QObject::connect(&client, &LlamaClient::requestFailed, &loop, [&](quint64, const QString &message) {
        netError = message;
        loop.quit();
    });
    QTimer::singleShot(120000, &loop, &QEventLoop::quit);

    const QString stored = QStringLiteral("[Вопрос с Хабра: Сигналы Qt | Раздел ответа: ответ 1, фрагмент 1/1]\n"
                                         "Сигнал connect связывают со слотом через QObject::connect.");
    client.requestEmbedding(stored);
    loop.exec();
    if (vector.isEmpty())
        return fail(out, netError.isEmpty() ? QStringLiteral("таймаут эмбеддинга") : netError);

    ChunkRecord record;
    record.sourceId = 1;
    record.question = QStringLiteral("Как связать сигнал и слот?");
    record.contextText = stored;
    record.answer = QStringLiteral("Через QObject::connect.");
    std::vector<float> stdVector(vector.cbegin(), vector.cend());
    if (!knowledge.addChunk(record, stdVector, &error))
        return fail(out, error);
    if (!knowledge.saveIndex(&error))
        return fail(out, error);
    if (knowledge.chunkCount() != 1 || knowledge.dimension() != vector.size())
        return fail(out, QStringLiteral("после записи база или размерность не сходятся"));

    const QVector<ScoredChunk> hits = knowledge.searchByVector(stdVector, 3, &error);
    if (!error.isEmpty())
        return fail(out, error);
    if (hits.isEmpty() || hits.first().chunk.id < 0)
        return fail(out, QStringLiteral("FAISS не вернул записанный чанк"));
    out << "faiss ok, dim=" << vector.size() << " score=" << hits.first().score << "\n";

    const QString fts = KnowledgeBase::buildFtsQuery(record.question);
    const QVector<ScoredChunk> textHits = knowledge.searchByText(record.question, 3, &error);
    if (!error.isEmpty())
        return fail(out, error);
    if (textHits.isEmpty())
        return fail(out, QStringLiteral("FTS5 ничего не нашёл по запросу: %1").arg(fts));
    out << "fts ok, query=" << fts << "\n";

    netError.clear();
    QString answer;
    QEventLoop chatLoop;
    QObject::connect(&client, &LlamaClient::chatReady, &chatLoop, [&](quint64, const QString &text) {
        answer = text;
        chatLoop.quit();
    });
    QObject::connect(&client, &LlamaClient::requestFailed, &chatLoop, [&](quint64, const QString &message) {
        netError = message;
        chatLoop.quit();
    });
    QTimer::singleShot(180000, &chatLoop, &QEventLoop::quit);
    client.requestChat(QStringLiteral("Ответь одним словом: пинг"), 32, 0.0);
    chatLoop.exec();
    if (answer.isEmpty())
        return fail(out, netError.isEmpty() ? QStringLiteral("таймаут генерации") : netError);

    out << "chat ok\nSELF-TEST OK\n";
    return 0;
}

} // namespace

int main(int argc, char *argv[])
{
    QApplication application(argc, argv);
    QCoreApplication::setOrganizationName(QStringLiteral("RAG"));
    QCoreApplication::setApplicationName(QStringLiteral("RAG"));

    if (application.arguments().contains(QStringLiteral("--self-test")))
        return runSelfTest();

    MainWindow window;
    window.show();
    return QApplication::exec();
}
