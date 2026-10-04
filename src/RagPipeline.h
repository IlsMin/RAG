#pragma once

#include "RagTypes.h"

#include <QJsonObject>
#include <QObject>
#include <QQueue>
#include <QVector>

#include <vector>

class KnowledgeBase;
class LlamaClient;

class RagPipeline : public QObject
{
    Q_OBJECT

public:
    RagPipeline(KnowledgeBase *knowledge, LlamaClient *client, QObject *parent = nullptr);

    void ask(const QString &question, const RagSettings &settings);
    void importJsonl(const QString &path, const RagSettings &settings);
    void cancel();

signals:
    void stepReady(const QString &title, const QString &body);
    void answerReady(const QString &answer);
    void errorOccurred(const QString &message);
    void importProgress(int done, int total);
    void importFinished(int chunksWritten);
    void busyChanged(bool busy);

private:
    struct PendingChunk {
        qint64 sourceId = 0;
        QString question;
        QString contextText;
        QString answer;
        int chunkIndex = 0;
    };

    enum class Phase {
        Idle,
        ImportEmbedding,
        MultiQuery,
        QueryEmbedding,
        Rerank,
        Answer,
    };

    void onEmbedding(quint64 requestId, const QVector<float> &vector);
    void onChat(quint64 requestId, const QString &text);
    void onFailed(quint64 requestId, const QString &error);

    void embedNextImportChunk();
    void finishImport();
    void requestMultiQuery();
    void beginQueryEmbeddings(const QStringList &queries);
    void embedNextQuery();
    void runSearch();
    void beginRerank();
    void beginAnswer(const QVector<ChunkRecord> &chunks);
    void fail(const QString &message);
    void becomeIdle();

    QVector<PendingChunk> chunksFromRecord(const QJsonObject &record) const;
    static QStringList parseVariants(const QString &text, int limit);
    static QVector<qint64> parseRankedIds(const QString &text);

    KnowledgeBase *m_knowledge = nullptr;
    LlamaClient *m_client = nullptr;
    RagSettings m_settings;
    Phase m_phase = Phase::Idle;
    bool m_cancelled = false;
    quint64 m_activeRequest = 0;

    QQueue<PendingChunk> m_importQueue;
    int m_importDone = 0;
    int m_importTotal = 0;

    QString m_question;
    int m_multiQueryAttempt = 0;
    QStringList m_queries;
    QVector<std::vector<float>> m_queryVectors;
    int m_embedPos = 0;
    QVector<ScoredChunk> m_fused;
    QVector<ChunkRecord> m_pinned;
    QVector<ChunkRecord> withPinned(const QVector<ChunkRecord> &chosen) const;
};
