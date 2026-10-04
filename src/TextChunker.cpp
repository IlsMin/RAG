#include "TextChunker.h"

#include <QRegularExpression>
#include <QTextDocument>

namespace TextChunker {
namespace {

QString normalize(QString text)
{
    text.replace(QStringLiteral("\r\n"), QStringLiteral("\n"));
    text.replace(u'\r', u'\n');
    text.replace(QChar(0x00A0), u' ');
    return text.trimmed();
}

QStringList splitKeep(const QString &text, const QRegularExpression &separator)
{
    QStringList parts;
    int last = 0;
    auto it = separator.globalMatch(text);
    while (it.hasNext()) {
        const auto match = it.next();
        const int end = match.capturedEnd();
        if (end <= last)
            continue;
        parts.append(text.mid(last, end - last));
        last = end;
    }
    if (last < text.size())
        parts.append(text.mid(last));

    QStringList filtered;
    for (const QString &part : parts) {
        if (!part.isEmpty())
            filtered.append(part);
    }
    return filtered;
}

QStringList hardWrap(const QString &text)
{
    if (text.size() <= kChunkSize)
        return {text};

    QStringList parts;
    int pos = 0;
    const QRegularExpression space(QStringLiteral("\\s"));
    while (pos < text.size()) {
        if (text.size() - pos <= kChunkSize) {
            parts.append(text.mid(pos));
            break;
        }
        int cut = pos + kChunkSize;
        const int spaceAt = text.lastIndexOf(space, cut);
        if (spaceAt > pos + kChunkSize / 2)
            cut = spaceAt + 1;
        parts.append(text.mid(pos, cut - pos));
        pos = cut;
    }
    return parts;
}

QStringList explode(const QString &text, const QList<QRegularExpression> &separators, int level)
{
    if (text.size() <= kChunkSize || level >= separators.size())
        return hardWrap(text);

    const QStringList parts = splitKeep(text, separators.at(level));
    if (parts.size() <= 1)
        return explode(text, separators, level + 1);

    QStringList fine;
    for (const QString &part : parts) {
        if (part.size() > kChunkSize)
            fine += explode(part, separators, level + 1);
        else
            fine.append(part);
    }
    return fine;
}

QStringList mergePieces(const QStringList &pieces)
{
    QStringList flat;
    for (const QString &piece : pieces) {
        if (piece.size() > kChunkSize)
            flat += hardWrap(piece);
        else if (!piece.isEmpty())
            flat.append(piece);
    }

    QStringList merged;
    QString current;
    for (const QString &piece : flat) {
        if (current.isEmpty()) {
            current = piece;
            continue;
        }
        if (current.size() + piece.size() <= kChunkSize)
            current += piece;
        else {
            merged.append(current);
            current = piece;
        }
    }
    if (!current.isEmpty())
        merged.append(current);
    return merged;
}

QString overlapTail(const QString &previous)
{
    if (kChunkOverlap <= 0 || previous.size() <= kChunkOverlap)
        return {};

    int start = previous.size() - kChunkOverlap;
    if (!previous.at(start).isSpace() && !previous.at(start - 1).isSpace()) {
        const QRegularExpression space(QStringLiteral("\\s"));
        const int forward = previous.indexOf(space, start);
        if (forward > start && forward < previous.size() - 1)
            start = forward + 1;
    }
    while (start < previous.size() && previous.at(start).isSpace())
        ++start;
    if (start >= previous.size())
        return {};
    return previous.mid(start);
}

QStringList applyOverlap(const QStringList &chunks)
{
    QStringList result;
    for (int i = 0; i < chunks.size(); ++i) {
        QString chunk = chunks.at(i);
        if (i > 0) {
            const QString tail = overlapTail(chunks.at(i - 1));
            if (!tail.isEmpty() && !chunk.startsWith(tail))
                chunk.prepend(tail);
        }
        chunk = chunk.trimmed();
        if (!chunk.isEmpty())
            result.append(chunk);
    }
    return result;
}

} // namespace

QString htmlToPlain(const QString &html)
{
    if (html.isEmpty())
        return {};
    if (!html.contains(u'<') && !html.contains(u'&'))
        return normalize(html);

    QTextDocument document;
    document.setHtml(html);
    return normalize(document.toPlainText());
}

QStringList split(const QString &text)
{
    const QString normalized = normalize(text);
    if (normalized.isEmpty())
        return {};
    if (normalized.size() <= kChunkSize)
        return {normalized};

    const QList<QRegularExpression> separators = {
        QRegularExpression(QStringLiteral("\\n\\n")),
        QRegularExpression(QStringLiteral("\\n")),
        QRegularExpression(QStringLiteral("[.!?…](?=\\s|$)")),
        QRegularExpression(QStringLiteral("\\s")),
    };
    return applyOverlap(mergePieces(explode(normalized, separators, 0)));
}

} // namespace TextChunker
