#include "LlamaClient.h"

#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QNetworkProxy>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QUrl>

namespace {
constexpr int kTimeoutMs = 180000;
}

LlamaClient::LlamaClient(QObject *parent)
    : QObject(parent)
{
    m_network.setProxy(QNetworkProxy::NoProxy);
}

QString LlamaClient::normalizeBase(QString url)
{
    url = url.trimmed();
    while (url.endsWith(u'/'))
        url.chop(1);

    const QStringList suffixes = {
        QStringLiteral("/v1/chat/completions"),
        QStringLiteral("/v1/embeddings"),
        QStringLiteral("/completion"),
        QStringLiteral("/embedding"),
    };
    for (const QString &suffix : suffixes) {
        if (url.endsWith(suffix)) {
            url.chop(suffix.size());
            break;
        }
    }
    while (url.endsWith(u'/'))
        url.chop(1);
    return url;
}

void LlamaClient::setChatBaseUrl(const QString &baseUrl)
{
    m_chatBase = normalizeBase(baseUrl);
}

void LlamaClient::setEmbeddingBaseUrl(const QString &baseUrl)
{
    m_embeddingBase = normalizeBase(baseUrl);
}

quint64 LlamaClient::requestEmbedding(const QString &text)
{
    QJsonObject body;
    body.insert(QStringLiteral("input"), text);
    const QByteArray payload = QJsonDocument(body).toJson(QJsonDocument::Compact);
    return postJson(m_embeddingBase + QStringLiteral("/v1/embeddings"), payload, QStringLiteral("embed"));
}

quint64 LlamaClient::requestChat(const QString &userPrompt, int maxTokens, double temperature,
                                     const QString &systemPrompt)
{
    QJsonArray messages;
    if (!systemPrompt.isEmpty()) {
        QJsonObject system;
        system.insert(QStringLiteral("role"), QStringLiteral("system"));
        system.insert(QStringLiteral("content"), systemPrompt);
        messages.append(system);
    }

    QJsonObject message;
    message.insert(QStringLiteral("role"), QStringLiteral("user"));
    message.insert(QStringLiteral("content"), userPrompt);
    messages.append(message);

    QJsonObject body;
    body.insert(QStringLiteral("model"), QStringLiteral("local"));
    body.insert(QStringLiteral("messages"), messages);
    body.insert(QStringLiteral("temperature"), temperature);
    body.insert(QStringLiteral("max_tokens"), maxTokens);
    body.insert(QStringLiteral("stream"), false);

    const QByteArray payload = QJsonDocument(body).toJson(QJsonDocument::Compact);
    return postJson(m_chatBase + QStringLiteral("/v1/chat/completions"), payload, QStringLiteral("chat"));
}

quint64 LlamaClient::postJson(const QString &url, const QByteArray &body, const QString &kind)
{
    const quint64 requestId = m_nextId++;
    QNetworkRequest request{QUrl(url)};
    request.setHeader(QNetworkRequest::ContentTypeHeader, QStringLiteral("application/json"));
    request.setTransferTimeout(kTimeoutMs);

    QNetworkReply *reply = m_network.post(request, body);
    reply->setProperty("requestId", requestId);
    reply->setProperty("kind", kind);
    reply->setProperty("url", url);
    m_replies.append(reply);
    connect(reply, &QNetworkReply::finished, this, [this, reply]() { onFinished(reply); });
    return requestId;
}

void LlamaClient::abort()
{
    const QList<QNetworkReply *> replies = m_replies;
    for (QNetworkReply *reply : replies)
        reply->abort();
}

void LlamaClient::onFinished(QNetworkReply *reply)
{
    m_replies.removeAll(reply);
    reply->deleteLater();

    const quint64 requestId = reply->property("requestId").toULongLong();
    const QString kind = reply->property("kind").toString();
    const QString url = reply->property("url").toString();

    if (reply->error() != QNetworkReply::NoError) {
        emit requestFailed(requestId, QStringLiteral("%1 — %2").arg(url, reply->errorString()));
        return;
    }

    const int status = reply->attribute(QNetworkRequest::HttpStatusCodeAttribute).toInt();
    const QByteArray payload = reply->readAll();
    if (status < 200 || status >= 300) {
        emit requestFailed(requestId,
                           QStringLiteral("HTTP %1: %2")
                               .arg(status)
                               .arg(QString::fromUtf8(payload.left(500))));
        return;
    }

    QJsonParseError parseError;
    const QJsonDocument document = QJsonDocument::fromJson(payload, &parseError);
    if (parseError.error != QJsonParseError::NoError || !document.isObject()) {
        emit requestFailed(requestId, QStringLiteral("Некорректный JSON от сервера Llama."));
        return;
    }
    const QJsonObject root = document.object();

    if (kind == QLatin1String("embed")) {
        const QJsonArray data = root.value(QStringLiteral("data")).toArray();
        if (data.isEmpty()) {
            emit requestFailed(requestId, QStringLiteral("В ответе эмбеддинга нет массива data."));
            return;
        }
        const QJsonArray values = data.at(0).toObject().value(QStringLiteral("embedding")).toArray();
        if (values.isEmpty()) {
            emit requestFailed(requestId, QStringLiteral("Пустой вектор эмбеддинга."));
            return;
        }
        QVector<float> vector;
        vector.reserve(values.size());
        for (const QJsonValue &value : values)
            vector.append(static_cast<float>(value.toDouble()));
        emit embeddingReady(requestId, vector);
        return;
    }

    const QJsonArray choices = root.value(QStringLiteral("choices")).toArray();
    if (choices.isEmpty()) {
        emit requestFailed(requestId, QStringLiteral("В ответе генерации нет choices."));
        return;
    }
    const QJsonObject choice = choices.at(0).toObject();
    QString text = choice.value(QStringLiteral("message")).toObject().value(QStringLiteral("content")).toString();
    if (text.isEmpty())
        text = choice.value(QStringLiteral("text")).toString();
    text = text.trimmed();
    if (text.isEmpty()) {
        emit requestFailed(requestId, QStringLiteral("Модель вернула пустой текст."));
        return;
    }
    emit chatReady(requestId, text);
}
