#ifndef WORKERTHREAD_H
#define WORKERTHREAD_H
#include <QThread>
#include "mainwindow.h"
#include "segmentation.h"

//derived class
class WorkerThread: public QThread { //public QThread makes WorkerThread a subclass of QThread
    Q_OBJECT // macro that declares its own signals and slots or that uses Qts meta-object system
public:
    //using QThread::QThread; //'using' allows us to adopt QThread as base constructor
    WorkerThread(QObject *parent = nullptr) : QThread(parent), pixmap(nullptr) {}
    ~WorkerThread(){delete pixmap;}

    void setData(QPixmap &p){
        delete pixmap;
        pixmap = new QPixmap(p);
    }

private:
    QPixmap *pixmap; //its a pointer because I want to be memory efficient

protected:
    void run() override {
        SegmentImg segImg;
        QPixmap seg_pixmap = segImg.segmentImage(pixmap);
        QString result = "Computation Complete.";
        emit resultReady(seg_pixmap, result);
    }

signals:
    void resultReady(const QPixmap &seg_pixmap, const QString &result);
};

#endif // WORKERTHREAD_H
