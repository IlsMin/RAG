#pragma once

#include "RagTypes.h"

#include <QObject>
#include <QSqlDatabase>
#include <QString>
#include <QVector>

#include <memory>
#include <vector>

namespace faiss {
struct Index;
}

class KnowledgeBase : public QObject
{
    Q_OBJECT

public:
    explicit KnowledgeBase(QObject *parent = nullptr);
    ~KnowledgeBase() override;

    bool open(const QString &directory, QString *error);
    QString directory() const;
    QString problem() const;
    int chunkCount();
    int dimension() const;

    bool addChunk(const ChunkRecord &chunk, const std::vector<float> &vector, QString *error);
    bool loadIndex(QString *error);
    bool saveIndex(QString *error);
    bool clear(QString *error);

    QVector<ScoredChunk> searchByVector(const std::vector<float> &vector, int k, QString *error);
    QVector<ScoredChunk> searchByText(const QString &query, int k, QString *error);
    static QString buildFtsQuery(const QString &text);

    ChunkRecord loadChunk(qint64 id);

private:
    bool createSchema(QString *error);
    bool ensureIndex(int dim, QString *error);
    void closeConnection();
    qint64 nextId();
    bool setMeta(const QString &key, const QString &value, QString *error);
    void refreshConsistency();

    QSqlDatabase m_db;
    QString m_directory;
    QString m_indexPath;
    QString m_problem;
    std::unique_ptr<faiss::Index> m_index;
};
