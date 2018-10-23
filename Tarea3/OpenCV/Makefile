#Compila Programa fuente

CC = g++
#CFLAGS1 = -pg -fopenmp
CFLAGS1 = -fpermissive
CFLAGS2 = -o
Archivo = tarea3_opencv
PATHBIN =
INCS1 = -I/usr/local/include/opencv
INCS2 = -I/usr/local/include/opencv2
INCS3 =
LIBS = -L/usr/local/lib -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_videoio -lm
PATHLIB =
MY_LIB =

All: binario

binario: $(Archivo).cpp
	$(CC) $(CFLAGS1) $(Archivo).cpp $(CFLAGS2) $(PATHBIN)$(Archivo) $(INCS1) $(INCS2) $(INCS3) $(LIBS)


clean:
		rm -rf *.o $(Archivo)
