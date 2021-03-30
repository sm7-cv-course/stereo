TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

unix {
INCLUDEPATH = /usr/local/include/opencv2/
#LIBS += -L/usr/local/lib -lopencv_stitching -lopencv_superres -lopencv_contrib -lopencv
PKGCONFIG += opencv
}
#unix: CONFIG += link_pkgconfig

SOURCES += \
        main.cpp
