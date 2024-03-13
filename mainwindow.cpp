#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "WorkerThread.h"
#include "segmentation.h"
#include <QFileDialog>
#include <QMessageBox>
#include <QPixmap>
#include <QImage>
#include <QVector>
#include <QQueue>
#include <cmath>
#include <iostream>
#include <Eigen/Dense>
#include <mlpack/methods/kmeans/kmeans.hpp>

//Function Prototypes
Eigen::MatrixXd convertToEigenMat(const QVector<QVector<int> > qtMat);


MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    // connecting browse button to ui
    connect(ui->pushButton_browse, &QPushButton::clicked, this, &MainWindow::on_pushButton_clicked);

    // connecting segment button to ui
    //connect(ui->pushButton_segment, &QPushButton::clicked, this, &MainWindow::segmentImage);
    connect(ui->pushButton_segment, &QPushButton::clicked, this, &MainWindow::handleSegmentation);

}

MainWindow::~MainWindow()
{
    delete ui;
}


// read and process image
void MainWindow::on_pushButton_clicked(){
    //opens dir and sets filename var equal to selected image file
    QString filename = QFileDialog::getOpenFileName(this, tr("Open File"), QDir::homePath(), tr("Images (*.png *.xpm *.jpg *.jpeg)"));

    if(!filename.isEmpty()){
        //open prompt and display image
        QMessageBox::information(this, "...", filename);
        QImage img(filename);
        QPixmap pix = QPixmap::fromImage(img);

        //get dimension info that is preset in mainwindow.ui
        int w = ui->labelMainImg->width();
        int h = ui->labelMainImg->height();

        //load img into ui
        ui->labelMainImg->setPixmap(pix.scaled(w,h,Qt::KeepAspectRatio));

        //get width/height, create empty binary matrix
        unsigned int cols = img.width();
        unsigned int rows = img.height();
        unsigned int numBlackPixels = 0;
        QVector<QVector<int>> imgArray(rows, QVector<int>(cols,0));


        // get pixel data, update matrix
        for (unsigned int i=0; i < rows; i++){
            for (unsigned int j = 0; j < cols; j++){
                // in .pixel(x,y) x = cols and y = rows;
                QColor clrCurrent(img.pixel(j,i));
                int r = clrCurrent.red();
                int g = clrCurrent.green();
                int b = clrCurrent.blue();
                int a = clrCurrent.alpha();
                //if black, assign 1 to array
                if (r+g+b < 20 && a > 240){
                    imgArray[i][j] = 1;
                    numBlackPixels += 1;
                }

            }
        }
        //store image array to a file (optional?)
        QString filename = "C:/Users/Michael Harris/Documents/test_qt";
        QFile fileout(filename);
        if(fileout.open(QFile::ReadWrite | QFile::Text)){
            QTextStream out(&fileout);

            for (unsigned int i = 0; i < rows; i++){
                for (unsigned int j = 0; j < cols; j++){
                    out << imgArray[i][j];
                }
                //end line after each col, and go to next row
                out << " " << Qt::endl;
            }
        }

        //update UI
        ui->labelData_dimensions->setText(QString::fromStdString("W: " + std::to_string(cols) + " H: " + std::to_string(rows)));


    }
}


void MainWindow::computationComplete(const QPixmap& resultPixmap){
    QLabel *label = ui->labelMainImg;
    label->setPixmap(resultPixmap);
    std::cout << "COMPUTATION COMPLETE" << std::endl;
}


void MainWindow::handleSegmentation(){
    //take info from the labelMainImg
    QLabel *label = ui->labelMainImg;
    QPixmap pixmap = label->pixmap();

    WorkerThread *workerThread = new WorkerThread(this);
    workerThread->setData(pixmap);

    connect(workerThread, &WorkerThread::resultReady, this, &computationComplete);
    connect(workerThread, &WorkerThread::finished, workerThread, &QObject::deleteLater); //calls delete on itself once computation is done
    workerThread->start(); //run() method is executed
}





