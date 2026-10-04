#pragma once

#include "RagTypes.h"

#include <QElapsedTimer>
#include <QMainWindow>

class KnowledgeBase;
class LlamaClient;
class RagPipeline;
class QCheckBox;
class QLabel;
class QLineEdit;
class QProgressBar;
class QPushButton;
class QSpinBox;
class QTextBrowser;

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow() override;

protected:
    void closeEvent(QCloseEvent *event) override;

private:
    void buildUi();
    void loadSettings();
    void saveSettings() const;
    RagSettings currentSettings() const;
    void appendChatBlock(const QString &html, bool separatorAfter);
    void appendStep(const QString &title, const QString &body);
    void setBusy(bool busy);
    void refreshStats();
    QString defaultJsonlPath() const;

    KnowledgeBase *m_knowledge = nullptr;
    LlamaClient *m_client = nullptr;
    RagPipeline *m_pipeline = nullptr;

    QTextBrowser *m_chat = nullptr;
    QTextBrowser *m_steps = nullptr;
    QLineEdit *m_question = nullptr;
    QPushButton *m_askButton = nullptr;

    QLineEdit *m_chatUrl = nullptr;
    QLineEdit *m_embedUrl = nullptr;
    QCheckBox *m_direct = nullptr;
    QCheckBox *m_multiQuery = nullptr;
    QCheckBox *m_hybrid = nullptr;
    QCheckBox *m_rerank = nullptr;
    QSpinBox *m_retrieveK = nullptr;
    QSpinBox *m_contextChunks = nullptr;
    QSpinBox *m_importLimit = nullptr;
    QLineEdit *m_jsonlPath = nullptr;
    QPushButton *m_importButton = nullptr;
    QPushButton *m_cancelButton = nullptr;
    QPushButton *m_clearButton = nullptr;
    QProgressBar *m_progress = nullptr;
    QLabel *m_stats = nullptr;
    QElapsedTimer m_askTimer;
    QString m_askTechniques;
};
