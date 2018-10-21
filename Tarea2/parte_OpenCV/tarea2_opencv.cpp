/* Autor: Alvaro Camacho Mora
  IIIC-2018
  Prof: Dr. Ing- Daniel Herrera Castro
  Tarea 2


  Este programa tiene como principal objetivo cambiar tres parametros de la camara :
  1- FrameRate
  2- Contraste
  3- Brillo

  Este codigo tiene como principal objetivo complementar los cambios hechos con el
  programa basado en ltilib-2. Este programa ofrece otros ejemplos como el cambio
  en el framerate, donde al disminuir se puede observar como los movimientos se ven
  menos naturales y parecen en camara lenta.

  Ademas al varias los otros parametros de la camara que se especifican en este ejemplo
  se puede obtener mejores resultados en cuando a calidad de la imagen. Por ejemplo,
  al reducir el brillo de la imagen se obtenia una definicion mas clara.

  Cabe destacar que para este problema no hubo ningun problema en cuanto a la calibracion
  inicial de la camara.
  */


#include "opencv2/opencv.hpp"
#include <iostream>

using namespace std;
using namespace cv;


void usage() {
  cout << " Con esta aplicacion podra hacer cambios dinamicamente a:\n";
  cout << "   -FrameRate (Max framerate: 30 fps, Min framerate: 5fps). Se dan incrementos/decrementos de 5fps, 25 fps no valido\n";
  cout << "   -Cambios en el contraste (incrementos/decrementos de 0.1)\n";
  cout << "   -Cambios en el brillo de la imagen (incrementos/decrementos de 0.1)\n ";
  cout << " Uso de la aplicacion:\n";
  cout << "   d/D : Disminuir framerate\n";
  cout << "   f/F : Increase framerate\n";
  cout << "   a/A : Aumentar el contraste \n";
  cout << "   s/S : Disminuir el contraste\n";
  cout << "   b/B : Aumentar el brillo \n";
  cout << "   n/N : Disminuir el brillo\n";
  cout << "   h/H : Uso de la aplicacion\n";
  cout << "   Esc : Salir de la aplicacion\n";
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


  VideoCapture image;
  if(!image.open(0)){  // si no se detecta la camara se cierra el programa
    cout << "Error! no se detecto ninguna camara!" << endl;
    return 0;
  }

// Se desplegan los valores actuales de la camara para que el usuario tenga una idea de ellos
  cout << "Valores actuales: " << endl;
  cout << "FrameRate: " << image.get(CAP_PROP_FPS) << endl;
  cout << "Brightness: " << image.get(CAP_PROP_BRIGHTNESS) << endl;
  cout << "Constraste: " << image.get(CAP_PROP_CONTRAST) << endl;
  double frame_value = 0;
  double contrast_value = 0;
  double brightness_value = 0;
  int c = 0;
  bool ok = true;
  Mat frame;

  while (ok)
  {

        image >> frame;
        if( frame.empty() )
          break;
        imshow("Imagen", frame);  //se proyecta el frame actual

        // se obtienen los valores actuales de las variables
        frame_value = image.get(CAP_PROP_FPS);
        contrast_value = image.get(CAP_PROP_CONTRAST);
        brightness_value = image.get(CAP_PROP_BRIGHTNESS);


        c = waitKey(100);   //se detecta la tecla presionada
        switch (c){
          case 27:  // si se detecta que se presiono un "Esc" se cierra el programa
            ok = false;
            break;
          // si se detecta que se presiono un d/D se incrementa el framerate
          case 70:
          case 102:
            if (frame_value<30){
              frame_value = frame_value + 5;
              if (frame_value == 25) frame_value = frame_value + 5;
              image.set(CAP_PROP_FPS,frame_value);
              cout << "FrameRate actual: " << image.get(CAP_PROP_FPS) << endl;
            }
            else {
              cout << "FrameRate actual: " << frame_value << endl;
            }
            break;
          // si se detecta que se presiono un f/F se disminuye el framerate
          case 68:
          case 100:
            if (frame_value>5){
              frame_value = frame_value - 5;
              if (frame_value == 25) frame_value = frame_value - 5;
              image.set(CAP_PROP_FPS,frame_value);
              cout << "FrameRate actual: " << image.get(CAP_PROP_FPS) << endl;
            }
            else {
              cout << "FrameRate actual: " << frame_value << endl;
            }
            break;
          // si se detecta que se presiono un a/A se incrementa el contraste
          case 65:
          case 97:
            if (contrast_value<1){
              contrast_value = contrast_value + 0.1094;
              image.set(CAP_PROP_CONTRAST,contrast_value);
              cout << "Contraste actual: " << image.get(CAP_PROP_CONTRAST) << endl;
            }
            else {
              cout << "Contraste actual: " << contrast_value << endl;
            }
            break;
          // si se detecta que se presiono un as/S se disminuye el contraste
          case 83:
          case 115:
            if (contrast_value>0){
              contrast_value = contrast_value - 0.1094;
              image.set(CAP_PROP_CONTRAST,contrast_value);
              cout << "Contraste actual: " << image.get(CAP_PROP_CONTRAST) << endl;
            }
            else {
              cout << "Contraste actual: " << contrast_value << endl;
            }
            break;
          // si se detecta que se presiono un b/B se incrementa el brillo
          case 66:
          case 98:
            if (brightness_value<1){
              brightness_value = brightness_value + 0.09375;
              cout << brightness_value << endl;
              image.set(CAP_PROP_BRIGHTNESS,brightness_value);
              cout << "Brillo actual: " << image.get(CAP_PROP_BRIGHTNESS) << endl;
            }
            else {
              cout << "Brillo actual: " << brightness_value << endl;
            }
            break;
          // si se detecta que se presiono un n/N se disminuye el brillo
          case 78:
          case 110:
            if (brightness_value>0){
              brightness_value = brightness_value - 0.09375;
              image.set(CAP_PROP_BRIGHTNESS,brightness_value);
              cout << "Brillo actual: " << image.get(CAP_PROP_BRIGHTNESS) << endl;
            }
            else {
              cout << "Brillo actual: " << brightness_value << endl;
            }
            break;
          // si se detecta que se presiono un h/H se imprime la ayuda para el uso del programa
          case 72:
          case 104:
            usage();
            break;
          default:
            break;
        }

  }

  return 0;
}
