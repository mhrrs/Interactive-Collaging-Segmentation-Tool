QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++17

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    main.cpp \
    mainwindow.cpp \
    segmentation.cpp

HEADERS += \
    WorkerThread.h \
    mainwindow.h \
    segmentation.h

FORMS += \
    mainwindow.ui

INCLUDEPATH += C:/C_libraries/eigen-3.4.0/eigen-3.4.0
INCLUDEPATH += "C:/Users/Michael Harris/vcpkg/installed/x64-windows/include"
INCLUDEPATH += C:/C_libraries/spectra-master/spectra-master/include

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

DISTFILES += \
    log.txt
