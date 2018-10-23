/* Autor: Alvaro Camacho Mora
IIIC-2018 Procesamiento Digital de Imagenes
Profesor: Dr. Daniel Herrera C.
Tarea3

Este programa realiza operaciones sobre los pixeles de la matriz de la imagen
para poder pintar una linea sobre la imagen utilizando el algoritmo de Bresenham.

Dentro de las principales caracteristicas estan que puede cargar imagenes de diferentes tipos:
  - Color
  - Escalares

La linea puede ser pintada de cualquier color (RGB con escala entre 0 y 255 para cada canal)
Estas intensidades en cada uno de los canales de color va a ser dada por el usuario por parametro

Ademas el punto de inicio y final tambien deben ser dados como parametro

Para mas detalles sobre el uso del programa se puede utilizar la opcion "-h"

*/



#include <opencv2/core/core.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;


void line(Mat& img, const Vec3b& color, Point& from, Point& to){

  int Dx, Dy, stepx, stepy, e, x, y;

  if (to.x > img.size().width) {
    cout << "Coordenada 'x' final se sale del rango maximo de la imagen. Ajustada al valor maximo de ancho (" << img.size().width << ")" << endl;
    to.x = img.size().width;
  }

  if (from.x > img.size().width) {
    cout << "Coordenada 'x' inicial se sale del rango maximo de la imagen. Ajustada al valor maximo de ancho (" << img.size().width << ")" << endl;
    from.x = img.size().width - 1;
  }

  if (to.y > img.size().height) {
    cout << "Coordenada 'y' final se sale del rango maximo de la imagen. Ajustada al valor maximo de alto (" << img.size().height << ")" << endl;
    to.y = img.size().height;
  }

  if (from.y > img.size().height) {
    cout << "Coordenada 'y' inicial se sale del rango maximo de la imagen. Ajustada al valor maximo de alto (" << img.size().height << ")" << endl;
    from.y = img.size().height - 1;
  }

  Dx = to.x - from.x;
  Dy = to.y - from.y;

  uint cuenta;

  if (Dy < 0) {
    Dy = -Dy;
    stepy = -1;
  } else {
    stepy = 1;
  }

  if (Dx < 0) {
    Dx = -Dx;
    stepx = -1;
    cuenta = Dx;
  }
  else if (Dx == 0){
    stepx = 0;
    cuenta = Dy;
  }
  else {
    stepx = 1;
    cuenta = Dx;
  }

  if ((Dx == 0) & (Dy == 0)) {
    cuenta = 1;
  }

  e = 2*Dy - Dx;
  x = from.x;
  y = from.y;

  for (uint i = 0; i < cuenta; i++) {
    img.at<Vec3b>(Point(x,y)) = color;
    x = x + stepx;
    if (e < 0) {
      e = e + 2*Dy;
    } else {
      y = y + stepy;
      e = e + 2*Dy - 2*Dx;
    }
  }
}

void usage() {
  cout << "Esta aplicacion permite dibujar una linea recta entre dos puntos dados" << endl;
  cout << "Los parametros necesarios deben ser:" << endl;
  cout << "    1- Imagen" << endl;
  cout << "    2- Intensidad de rojo  (R)" << endl;
  cout << "    3- Intensidad de verde (G)" << endl;
  cout << "    4- Intensidad de azul  (B)" << endl;
  cout << "    5- Coordenada 'x' para el punto inicial" << endl;
  cout << "    6- Coordenada 'y' para el punto inicia" << endl;
  cout << "    7- Coordenada 'x' para el punto inicia" << endl;
  cout << "    8- Coordenada 'y' para el punto inicia" << endl;
  cout << "Ejemplo de parametros: imagen1.jpg 255 0 0 50 100 60 80" << endl;
  cout << "Con el ejemplo anterior se estaria dibujando una linea roja entre los puntos (50,100) y (60,80)" << endl;
  std::cout << "******* Si alguno de los puntos ingresados es mayor al tamano de la imagen, se reasignaran a los  valores maximos de la imagen *****" << '\n';
}

/*
 * Parse the line command arguments
 */
void parseArgs(int argc, char*argv[]) {

  // check each argument of the command line
  for (int i=1; i<argc; i++) {
    if (*argv[i] == '-') {
      switch (argv[i][1]) {
      case 'h':
        usage();
        exit(EXIT_SUCCESS);
        break;
      default:
        break;
      }
    }
  }
}

int main(int argc, char** argv)
{

  parseArgs(argc,argv); // se revisan los argumentos

    if( argc != 9)
    {
     cout <<" Faltan argumentos" << endl;
     return -1;
    }

    Vec3b color;
    Point from, to;
    Mat img;
    img = imread(argv[1], CV_LOAD_IMAGE_COLOR | CV_LOAD_IMAGE_ANYDEPTH);

    color[2] = atoi(argv[2]);
    color[1] = atoi(argv[3]);
    color[0] = atoi(argv[4]);

    from.x = atoi(argv[5]);
    from.y = atoi(argv[6]);

    to.x = atoi(argv[7]);
    to.y = atoi(argv[8]);

    line(img, color, from, to);

    if(! img.data )
    {
        cout <<  "Imagen no encontrada" << std::endl ;
        return -1;
    }

    namedWindow( "Imagen", WINDOW_NORMAL);
    imshow( "Imagen", img );


    waitKey(0);
    return 0;
}
