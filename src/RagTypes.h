#pragma once

#include <QString>
#include <QtGlobal>

struct ChunkRecord {
    qint64 id = -1;
    qint64 sourceId = 0;
    QString question;
    QString contextText;
    QString answer;
    int chunkIndex = 0;
};

struct ScoredChunk {
    ChunkRecord chunk;
    float score = 0.f;
    QString via;
};

struct RagSettings {
    QString chatBaseUrl = QStringLiteral("http://127.0.0.1:8080");
    QString embeddingBaseUrl = QStringLiteral("http://127.0.0.1:8081");
    bool direct = false;
    bool multiQuery = true;
    bool hybridSearch = true;
    bool rerank = true;
    int retrieveK = 8;
    int contextChunks = 2;
    int importLimit = 200;
};
