#-------------------------------------------------
#
# Project created by QtCreator 2012-12-10T11:53:40
#
#-------------------------------------------------

QT       += core

QT       -= gui

TARGET = tracking
CONFIG   += console
CONFIG   -= app_bundle

TEMPLATE = app


SOURCES += main.cpp

INCLUDEPATH += /Users/bingeb/opencv243/include
LIBS += -L/Users/bingeb/opencv243/lib \
-lopencv_core \
-lopencv_highgui \
-lopencv_imgproc \
-lopencv_features2d \
-lopencv_ml \
-lopencv_video \
-lopencv_objdetect
