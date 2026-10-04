#pragma once

#include <QString>
#include <QStringList>

namespace TextChunker {

inline constexpr int kChunkSize = 600;
inline constexpr int kChunkOverlap = 120;

QString htmlToPlain(const QString &html);
QStringList split(const QString &text);

} // namespace TextChunker
