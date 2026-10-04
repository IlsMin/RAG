#pragma once

#include <QList>
#include <QNetworkAccessManager>
#include <QObject>
#include <QString>
#include <QVector>

class QNetworkReply;

class LlamaClient : public QObject
{
    Q_OBJECT

public:
    explicit LlamaClient(QObject *parent = nullptr);

    void setChatBaseUrl(const QString &baseUrl);
    void setEmbeddingBaseUrl(const QString &baseUrl);

    quint64 requestEmbedding(const QString &text);
    quint64 requestChat(const QString &userPrompt, int maxTokens, double temperature,
                        const QString &systemPrompt = {});
    void abort();

signals:
    void embeddingReady(quint64 requestId, QVector<float> vector);
    void chatReady(quint64 requestId, QString text);
    void requestFailed(quint64 requestId, QString error);

private:
    quint64 postJson(const QString &url, const QByteArray &body, const QString &kind);
    void onFinished(QNetworkReply *reply);
    static QString normalizeBase(QString url);

    QNetworkAccessManager m_network;
    QString m_chatBase = QStringLiteral("http://127.0.0.1:8080");
    QString m_embeddingBase = QStringLiteral("http://127.0.0.1:8081");
    quint64 m_nextId = 1;
    QList<QNetworkReply *> m_replies;
};
