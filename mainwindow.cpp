#include "mainwindow.h"

#include "KnowledgeBase.h"
#include "LlamaClient.h"
#include "RagPipeline.h"

#include <algorithm>

#include <QCheckBox>
#include <QCloseEvent>
#include <QCoreApplication>
#include <QDir>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QProgressBar>
#include <QPushButton>
#include <QSettings>
#include <QSpinBox>
#include <QSplitter>
#include <QStringList>
#include <QStatusBar>
#include <QTabWidget>
#include <QTextBlockFormat>
#include <QTextBrowser>
#include <QTextCursor>
#include <QVBoxLayout>

namespace {
QString htmlEscape(const QString &text)
{
    return text.toHtmlEscaped();
}

QString formatElapsed(qint64 milliseconds)
{
    const double seconds = milliseconds / 1000.0;
    if (seconds < 60.0)
        return QStringLiteral("%1 с").arg(QString::number(seconds, 'f', 1));
    const int whole = static_cast<int>(seconds);
    return QStringLiteral("%1 мин %2 с").arg(whole / 60).arg(whole % 60);
}

QString enabledTechniques(bool direct, bool multiQuery, bool hybridSearch, bool rerank)
{
    if (direct)
        return QStringLiteral("без RAG");
    QStringList names;
    if (multiQuery)
        names << QStringLiteral("multi-query");
    if (hybridSearch)
        names << QStringLiteral("hybrid");
    if (rerank)
        names << QStringLiteral("rerank");
    return names.join(QStringLiteral(", "));
}
}

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , m_knowledge(new KnowledgeBase(this))
    , m_client(new LlamaClient(this))
    , m_pipeline(new RagPipeline(m_knowledge, m_client, this))
{
    buildUi();
    loadSettings();

    const QString dataDir = QDir(QCoreApplication::applicationDirPath()).filePath(QStringLiteral("data"));
    QString error;
    if (!m_knowledge->open(dataDir, &error)) {
        m_stats->setText(error);
        QMessageBox::warning(this, QStringLiteral("RAG"), error);
    }
    refreshStats();

    connect(m_pipeline, &RagPipeline::stepReady, this, &MainWindow::appendStep);
    connect(m_pipeline, &RagPipeline::answerReady, this, [this](const QString &answer) {
        QString timing = QStringLiteral("<b>Время: </b><i>%1</i>")
                             .arg(htmlEscape(formatElapsed(m_askTimer.elapsed())));
        if (!m_askTechniques.isEmpty())
            timing += QStringLiteral(" (%1)").arg(htmlEscape(m_askTechniques));
        appendChatBlock(QStringLiteral("<b>Руководство</b><br>%1<br>%2")
                             .arg(htmlEscape(answer), timing),
                         true);
    });
    connect(m_pipeline, &RagPipeline::errorOccurred, this, [this](const QString &message) {
        appendStep(QStringLiteral("Ошибка"), message);
        QMessageBox::warning(this, QStringLiteral("RAG"), message);
        setBusy(false);
        refreshStats();
    });
    connect(m_pipeline, &RagPipeline::busyChanged, this, &MainWindow::setBusy);
    connect(m_pipeline, &RagPipeline::importProgress, this, [this](int done, int total) {
        m_progress->setRange(0, std::max(1, total));
        m_progress->setValue(done);
        statusBar()->showMessage(QStringLiteral("Импорт: %1 / %2 чанков").arg(done).arg(total));
    });
    connect(m_pipeline, &RagPipeline::importFinished, this, [this](int) {
        refreshStats();
        statusBar()->showMessage(QStringLiteral("В базе %1 чанков").arg(m_knowledge->chunkCount()), 5000);
    });

    connect(m_askButton, &QPushButton::clicked, this, [this]() {
        saveSettings();
        const QString question = m_question->text().trimmed();
        if (question.isEmpty()) {
            QMessageBox::warning(this, QStringLiteral("RAG"), QStringLiteral("Введите вопрос."));
            return;
        }
        m_steps->clear();
        appendChatBlock(QStringLiteral("<b>Вы</b><br>%1").arg(htmlEscape(question)), false);
        m_client->setChatBaseUrl(m_chatUrl->text());
        m_client->setEmbeddingBaseUrl(m_embedUrl->text());
        m_askTimer.start();
        m_askTechniques = enabledTechniques(
            m_direct->isChecked(), m_multiQuery->isChecked(), m_hybrid->isChecked(), m_rerank->isChecked());
        m_pipeline->ask(question, currentSettings());
    });
    connect(m_question, &QLineEdit::returnPressed, m_askButton, &QPushButton::click);

    connect(m_importButton, &QPushButton::clicked, this, [this]() {
        saveSettings();
        m_steps->clear();
        statusBar()->showMessage(QStringLiteral("Чтение jsonl…"));
        m_pipeline->importJsonl(m_jsonlPath->text().trimmed(), currentSettings());
    });
    connect(m_cancelButton, &QPushButton::clicked, m_pipeline, &RagPipeline::cancel);
    connect(m_clearButton, &QPushButton::clicked, this, [this]() {
        const auto answer = QMessageBox::question(
            this,
            QStringLiteral("Очистить базу"),
            QStringLiteral("Удалить все чанки и векторный индекс?"));
        if (answer != QMessageBox::Yes)
            return;
        QString error;
        if (!m_knowledge->clear(&error)) {
            QMessageBox::warning(this, QStringLiteral("RAG"), error);
            return;
        }
        refreshStats();
        statusBar()->showMessage(QStringLiteral("База очищена"), 4000);
    });
}

MainWindow::~MainWindow() = default;

void MainWindow::buildUi()
{
    setWindowTitle(QStringLiteral("RAG — база знаний Хабра"));
    resize(1180, 740);

    auto *tabs = new QTabWidget(this);
    setCentralWidget(tabs);

    auto *chatPage = new QWidget(tabs);
    auto *chatLayout = new QVBoxLayout(chatPage);
    auto *splitter = new QSplitter(Qt::Horizontal, chatPage);

    auto *left = new QWidget(splitter);
    auto *leftLayout = new QVBoxLayout(left);
    leftLayout->setContentsMargins(0, 0, 0, 0);
    m_chat = new QTextBrowser(left);
    m_chat->setOpenExternalLinks(false);
    auto *askRow = new QHBoxLayout();
    m_question = new QLineEdit(left);
    m_question->setPlaceholderText(QStringLiteral("Вопрос к базе знаний"));
    m_askButton = new QPushButton(QStringLiteral("Спросить"), left);
    askRow->addWidget(m_question, 1);
    askRow->addWidget(m_askButton);
    leftLayout->addWidget(m_chat, 1);
    leftLayout->addLayout(askRow);

    auto *right = new QWidget(splitter);
    auto *rightLayout = new QVBoxLayout(right);
    rightLayout->setContentsMargins(0, 0, 0, 0);
    rightLayout->addWidget(new QLabel(QStringLiteral("Шаги поиска"), right));
    m_steps = new QTextBrowser(right);
    m_steps->setOpenExternalLinks(false);
    rightLayout->addWidget(m_steps, 1);

    splitter->addWidget(left);
    splitter->addWidget(right);
    splitter->setStretchFactor(0, 3);
    splitter->setStretchFactor(1, 2);
    chatLayout->addWidget(splitter);
    tabs->addTab(chatPage, QStringLiteral("Диалог"));

    auto *settingsPage = new QWidget(tabs);
    auto *settingsLayout = new QVBoxLayout(settingsPage);
    auto *form = new QFormLayout();
    m_chatUrl = new QLineEdit(settingsPage);
    m_embedUrl = new QLineEdit(settingsPage);
    m_direct = new QCheckBox(QStringLiteral("Без RAG — вопрос сразу в модель, без поиска по базе"), settingsPage);
    m_multiQuery = new QCheckBox(QStringLiteral("Multi-query — несколько формулировок вопроса"), settingsPage);
    m_hybrid = new QCheckBox(QStringLiteral("Hybrid search — FAISS и полнотекст FTS5"), settingsPage);
    m_rerank = new QCheckBox(QStringLiteral("Rerank — модель заново упорядочивает чанки"), settingsPage);
    m_retrieveK = new QSpinBox(settingsPage);
    m_retrieveK->setRange(1, 20);
    m_contextChunks = new QSpinBox(settingsPage);
    m_contextChunks->setRange(1, 4);
    m_importLimit = new QSpinBox(settingsPage);
    m_importLimit->setRange(1, 5000);
    form->addRow(QStringLiteral("URL чата"), m_chatUrl);
    form->addRow(QStringLiteral("URL эмбеддингов"), m_embedUrl);
    form->addRow(QString(), m_direct);
    form->addRow(QString(), m_multiQuery);
    form->addRow(QString(), m_hybrid);
    form->addRow(QString(), m_rerank);
    form->addRow(QStringLiteral("Кандидатов на список"), m_retrieveK);
    form->addRow(QStringLiteral("Чанков в финальном промпте"), m_contextChunks);
    form->addRow(QStringLiteral("Лимит вопросов при импорте"), m_importLimit);
    settingsLayout->addLayout(form);

    auto *fileRow = new QHBoxLayout();
    m_jsonlPath = new QLineEdit(settingsPage);
    auto *browse = new QPushButton(QStringLiteral("Обзор…"), settingsPage);
    fileRow->addWidget(m_jsonlPath, 1);
    fileRow->addWidget(browse);
    settingsLayout->addWidget(new QLabel(QStringLiteral("Файл questions.jsonl"), settingsPage));
    settingsLayout->addLayout(fileRow);
    connect(browse, &QPushButton::clicked, this, [this]() {
        const QString path = QFileDialog::getOpenFileName(
            this,
            QStringLiteral("База знаний"),
            QFileInfo(m_jsonlPath->text()).absolutePath(),
            QStringLiteral("JSON Lines (*.jsonl);;Все файлы (*.*)"));
        if (!path.isEmpty())
            m_jsonlPath->setText(path);
    });

    auto *buttonRow = new QHBoxLayout();
    m_importButton = new QPushButton(QStringLiteral("Импорт"), settingsPage);
    m_cancelButton = new QPushButton(QStringLiteral("Остановить"), settingsPage);
    m_clearButton = new QPushButton(QStringLiteral("Очистить базу"), settingsPage);
    m_cancelButton->setEnabled(false);
    buttonRow->addWidget(m_importButton);
    buttonRow->addWidget(m_cancelButton);
    buttonRow->addWidget(m_clearButton);
    buttonRow->addStretch(1);
    settingsLayout->addLayout(buttonRow);

    m_progress = new QProgressBar(settingsPage);
    m_progress->setRange(0, 1);
    m_progress->setValue(0);
    m_progress->setTextVisible(true);
    settingsLayout->addWidget(m_progress);
    m_stats = new QLabel(settingsPage);
    m_stats->setWordWrap(true);
    settingsLayout->addWidget(m_stats);
    settingsLayout->addStretch(1);
    tabs->addTab(settingsPage, QStringLiteral("Настройки"));

    connect(m_direct, &QCheckBox::toggled, this, [this](bool direct) {
        m_multiQuery->setEnabled(!direct);
        m_hybrid->setEnabled(!direct);
        m_rerank->setEnabled(!direct);
        m_retrieveK->setEnabled(!direct);
        m_contextChunks->setEnabled(!direct);
    });

    statusBar()->showMessage(QStringLiteral("Чат :8080, эмбеддинги :8081"));
}

void MainWindow::loadSettings()
{
    QSettings settings;
    m_chatUrl->setText(settings.value(QStringLiteral("chatUrl"), QStringLiteral("http://127.0.0.1:8080")).toString());
    m_embedUrl->setText(settings.value(QStringLiteral("embedUrl"), QStringLiteral("http://127.0.0.1:8081")).toString());
    m_direct->setChecked(settings.value(QStringLiteral("direct"), false).toBool());
    m_multiQuery->setChecked(settings.value(QStringLiteral("multiQuery"), true).toBool());
    m_hybrid->setChecked(settings.value(QStringLiteral("hybrid"), true).toBool());
    m_rerank->setChecked(settings.value(QStringLiteral("rerank"), true).toBool());
    m_retrieveK->setValue(settings.value(QStringLiteral("retrieveK"), 8).toInt());
    m_contextChunks->setValue(settings.value(QStringLiteral("contextChunks"), 2).toInt());
    m_importLimit->setValue(settings.value(QStringLiteral("importLimit"), 200).toInt());
    m_jsonlPath->setText(settings.value(QStringLiteral("jsonlPath"), defaultJsonlPath()).toString());
    m_direct->toggled(m_direct->isChecked());
}

void MainWindow::saveSettings() const
{
    QSettings settings;
    settings.setValue(QStringLiteral("chatUrl"), m_chatUrl->text().trimmed());
    settings.setValue(QStringLiteral("embedUrl"), m_embedUrl->text().trimmed());
    settings.setValue(QStringLiteral("direct"), m_direct->isChecked());
    settings.setValue(QStringLiteral("multiQuery"), m_multiQuery->isChecked());
    settings.setValue(QStringLiteral("hybrid"), m_hybrid->isChecked());
    settings.setValue(QStringLiteral("rerank"), m_rerank->isChecked());
    settings.setValue(QStringLiteral("retrieveK"), m_retrieveK->value());
    settings.setValue(QStringLiteral("contextChunks"), m_contextChunks->value());
    settings.setValue(QStringLiteral("importLimit"), m_importLimit->value());
    settings.setValue(QStringLiteral("jsonlPath"), m_jsonlPath->text().trimmed());
}

RagSettings MainWindow::currentSettings() const
{
    RagSettings settings;
    settings.chatBaseUrl = m_chatUrl->text().trimmed();
    settings.embeddingBaseUrl = m_embedUrl->text().trimmed();
    settings.direct = m_direct->isChecked();
    settings.multiQuery = m_multiQuery->isChecked();
    settings.hybridSearch = m_hybrid->isChecked();
    settings.rerank = m_rerank->isChecked();
    settings.retrieveK = m_retrieveK->value();
    settings.contextChunks = m_contextChunks->value();
    settings.importLimit = m_importLimit->value();
    return settings;
}

void MainWindow::appendChatBlock(const QString &html, bool separatorAfter)
{
    QTextCursor cursor(m_chat->document());
    cursor.movePosition(QTextCursor::End);
    if (!m_chat->document()->isEmpty())
        cursor.insertBlock(QTextBlockFormat());
    cursor.insertHtml(html);
    if (separatorAfter) {
        QTextBlockFormat rule;
        rule.setProperty(QTextFormat::BlockTrailingHorizontalRulerWidth, 1);
        cursor.mergeBlockFormat(rule);
    }
    m_chat->setTextCursor(cursor);
    m_chat->ensureCursorVisible();
}

void MainWindow::appendStep(const QString &title, const QString &body)
{
    m_steps->append(QStringLiteral("<b>%1</b>").arg(htmlEscape(title)));
    m_steps->append(QStringLiteral("<pre>%1</pre>").arg(htmlEscape(body)));
}

void MainWindow::setBusy(bool busy)
{
    m_askButton->setEnabled(!busy);
    m_question->setEnabled(!busy);
    m_importButton->setEnabled(!busy);
    m_clearButton->setEnabled(!busy);
    m_cancelButton->setEnabled(busy);
    if (!busy)
        statusBar()->showMessage(QStringLiteral("В базе %1 чанков").arg(m_knowledge->chunkCount()), 4000);
}

void MainWindow::refreshStats()
{
    if (!m_knowledge->problem().isEmpty()) {
        m_stats->setText(m_knowledge->problem());
        return;
    }
    const int dim = m_knowledge->dimension();
    m_stats->setText(QStringLiteral("Чанков в базе: %1. Размерность FAISS: %2. Каталог: %3")
                         .arg(m_knowledge->chunkCount())
                         .arg(dim > 0 ? QString::number(dim) : QStringLiteral("появится после первого чанка"))
                         .arg(m_knowledge->directory()));
}

QString MainWindow::defaultJsonlPath() const
{
    const QStringList candidates = {
        QDir::current().filePath(QStringLiteral("questions.jsonl")),
        QDir(QCoreApplication::applicationDirPath()).filePath(QStringLiteral("questions.jsonl")),
        QStringLiteral("C:/Projects/wrk_CURSOR/RAG/questions.jsonl"),
    };
    for (const QString &path : candidates) {
        if (QFileInfo::exists(path))
            return path;
    }
    return candidates.constLast();
}

void MainWindow::closeEvent(QCloseEvent *event)
{
    saveSettings();
    m_pipeline->cancel();
    QString error;
    m_knowledge->saveIndex(&error);
    QMainWindow::closeEvent(event);
}
