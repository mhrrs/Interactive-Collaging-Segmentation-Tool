#ifndef SEGMENTATION_H
#define SEGMENTATION_H
#include <QPixmap>
#include <QImage>
#include <QVector>
#include <Eigen/Dense>

class SegmentImg{
public:
    //Primary Functions
    QPixmap segmentImage(QPixmap *pixels);

    //covert to grayscale
    QPixmap convertToGrayscale(const QPixmap &original);

    //resize img, construct and return adjacency matrix
    QVector<QVector<double>> createAdjMatrix(QImage image, unsigned int &rows, unsigned int &cols, unsigned int &w, unsigned int &h);
    QVector<QVector<double>> computeEpsilonNearestNeighbors(const QImage& image, double epsilon, int w, int h);

    //create diagonal matrix
    QVector<QVector<double>> createDegreeMatrix(QVector<QVector<double>> &adjMatrix);

    //compute degree matrix and return the laplacian matrix
    QVector<QVector<double>> createLaplacianMatrix(QVector<QVector<double>> &adjMatrix, QVector<QVector<double>> &degMatrix);
};

#endif // SEGMENTATION_H
