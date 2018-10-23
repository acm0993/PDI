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


// Standard Headers: from ANSI C and GNU C Library
#include <cstdlib>  // Standard Library for C++
#include <getopt.h> // Functions to parse the command line arguments

// Standard Headers: STL
#include <iostream>
#include <string>
#include <fstream>

// LTI-Lib Headers
#include <ltiObject.h>
#include <ltiMath.h>     // General lti:: math and <cmath> functionality
#include <ltiTimer.h>    // To measure time

#include <ltiIOImage.h> // To read/write images from files (jpg, png, and bmp)
#include <ltiViewer2D.h>

#include <ltiLispStreamHandler.h>

// Ensure that the STL streaming is used.
using std::cout;
using std::cerr;
using std::endl;


/*
 * Help
 */
 void usage() {
   cout << "Esta aplicacion permite dibujar una linea recta entre dos puntos dados" << endl;
   cout << "Los parametros necesarios deben ser:" << endl;
   cout << "    1- Imagen" << endl;
   cout << "    2- Intensidad de rojo  (R)" << endl;
   cout << "    3- Intensidad de verde (G)" << endl;
   cout << "    4- Intensidad de azul  (B)" << endl;
   cout << "    5- Coordenada 'x' para el punto inicial" << endl;
   cout << "    6- Coordenada 'y' para el punto inicial" << endl;
   cout << "    7- Coordenada 'x' para el punto inicial" << endl;
   cout << "    8- Coordenada 'y' para el punto inicial" << endl;
   cout << "Ejemplo de parametros: imagen1.jpg 255 0 0 50 100 60 80" << endl;
   cout << "Con el ejemplo anterior se estaria dibujando una linea roja entre los puntos (50,100) y (60,80)" << endl;
   std::cout << "******* Si alguno de los puntos ingresados es mayor al tamano de la imagen, se reasignaran a los  valores maximos de la imagen *****" << '\n';
 }


void parse(int argc, char*argv[],std::string& filename) {

  int c;

  // We use the standard getopt.h functions here to parse the arguments.
  // Check the documentation of getopt.h for more information on this

  // structure for the long options.
  static struct option lopts[] = {
    {"help",no_argument,0,'h'},
    {0,0,0,0}
  };

  int optionIdx;

  while ((c = getopt_long(argc, argv, "h", lopts,&optionIdx)) != -1) {
    switch (c) {
    case 'h':
      usage();
      exit(EXIT_SUCCESS);
    default:
      cerr << "Option '-" << static_cast<char>(c) << "' not recognized."
           << endl;
    }
  }

  // Now try to read the image name
  if (optind < argc) {           // if there are still some arguments left...
    filename = argv[optind];  // with the given image file
  } else {
    filename = "";
  }
}

template <typename T>

void line(lti::matrix<T>& img, const T& color, lti::ipoint& from, lti::ipoint& to){

  int Dx, Dy, stepx, stepy, e, x, y;

  if (to.x > img.columns()) {
    cout << "Coordenada 'x' final se sale del rango maximo de la imagen. Ajustada al valor maximo de ancho (" << img.rows() << ")" << endl;
    to.x = img.columns();
  }

  if (from.x > img.columns()) {
    cout << "Coordenada 'x' inicial se sale del rango maximo de la imagen. Ajustada al valor maximo de ancho (" << img.rows() << ")" << endl;
    from.x = img.columns() - 1;
  }

  if (to.y > img.rows()) {
    cout << "Coordenada 'y' final se sale del rango maximo de la imagen. Ajustada al valor maximo de alto (" << img.columns() << ")" << endl;
    to.y = img.rows();
  }

  if (from.y > img.rows()) {
    cout << "Coordenada 'y' inicial se sale del rango maximo de la imagen. Ajustada al valor maximo de alto (" << img.columns() << ")" << endl;
    from.y = img.rows() - 1;
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
    img.at(lti::ipoint(x,y)) = color;
    x = x + stepx;
    if (e < 0) {
      e = e + 2*Dy;
    } else {
      y = y + stepy;
      e = e + 2*Dy - 2*Dx;
    }
  }
}

int main(int argc, char* argv[]) {

  // check if the user gave some filename
  std::string filename;
  parse(argc,argv,filename);

  if( argc != 9)
  {
   cout <<" Faltan argumentos" << endl;
   return -1;
  }

  if (!filename.empty()) {
    // we need an object to load images
    lti::ioImage loader;
    // we also need an image object
    lti::image img;
    // load the image
    if (loader.load(filename,img)) {
      static lti::viewer2D view;
      lti::rgbaPixel color;
      lti::ipoint from, to;

      from = lti::ipoint(atoi(argv[5]),atoi(argv[6]));
      to = lti::ipoint(atoi(argv[7]),atoi(argv[8]));

      color.set(atoi(argv[2]),atoi(argv[3]),atoi(argv[4]),0);

      line(img,color,from,to);

      lti::viewer2D::parameters vpar(view.getParameters());
      vpar.title = filename; // set the image name at the window's title bar
      view.setParameters(vpar);

      view.show(img); // show the image

      // wait for the user to close the window or to indicate
      bool theEnd = false;
      lti::viewer2D::interaction action;
      lti::ipoint pos;

      do {
        view.waitInteraction(action,pos); // wait for something to happen
        if (action == lti::viewer2D::Closed) { // window closed?
          theEnd = true; // we are ready here!
        }
      } while(!theEnd);


      return EXIT_SUCCESS;
    }
    return EXIT_FAILURE;

  } else {
    cout << "Not done yet!  Try giving a image name" << endl;
    usage();

    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}
