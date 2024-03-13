#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    //Primary Functions:
    void on_pushButton_clicked();
    void handleSegmentation();
    void computationComplete(const QPixmap& resultPixmap);



private:
    Ui::MainWindow *ui;
};
#endif // MAINWINDOW_H
