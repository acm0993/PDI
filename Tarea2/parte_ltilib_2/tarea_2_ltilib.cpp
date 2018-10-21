/* Autor: Alvaro Camacho Mora
  IIIC-2018
  Prof: Dr. Ing- Daniel Herrera Castro
  Tarea 2
  Basado en el ejemplo "v4l2"


  Este programa tiene como principal objetivo cambiar tres parametros de la camara :
  1- Saturacion
  2- Matiz
  3- Brillo

  Ademas, se tuvieron problemas con obtener una imagen nitida por parte de la camra, estos
  fueron sulucionados revisando la documentacion de la biblioteca para entender cada uno de
  los parametros de la camara. Con el archivo lsp adjunto estos problemas fueron resueltos.

  De los resultados mas interesantes obtenidos es el cambio de coloracion y difinicion con
  la que la imagen se ve al cambiar todos los parametros. Incluso al cambiar de cierta
  forma los parametros de este trabajo se dan coloraciones rojizas y verduzcas en la
  imagen resultante.

  */

// LTI-Lib Headers
#include "ltiObject.h"
#include "ltiMath.h"     // General lti:: math and <cmath> functionality
#include "ltiConfig.h"   // To check if GTK is there
#include "ltiTimer.h"    // To measure time

#include "ltiChannel.h" // Monochromatic byte channels
#include "ltiImage.h"


#include "ltiLispStreamHandler.h"


#include "ltiV4l2.h"
#include "ltiV4l2_patch.h"
#include "ltiViewer2D.h" // The normal viewer

// Standard Headers
#include <cstdlib>
#include <iostream>
#include <string>
#include <fstream>

using std::cout;
using std::cerr;
using std::endl;


void usage() {
  cout << " Con esta aplicacion podra hacer cambios dinamicamente a:\n";
  cout << "   - Cambios en la saturacion (incrementos/decrementos de 0.05)\n";
  cout << "   - Cambios en el matiz (Hue) (incrementos/decrementos de 0.05)\n";
  cout << "   - Cambios en el brillo (incrementos/decrementos de 0.05)\n ";
  cout << " Uso de la aplicacion:\n";
  cout << "   d/D : Disminuir saturacion\n";
  cout << "   f/F : Increase saturacion\n";
  cout << "   a/A : Aumentar el matiz \n";
  cout << "   s/S : Disminuir el matiz\n";
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

/*
 * Main method
 */
int main(int argc, char* argv[]) {

  parseArgs(argc,argv); // se revisan los parametros introducidos por el usuario

  static const char* confFile = "parametros_camara.lsp"; // se carga el archivo de configuracion de la camara



  lti::v4l2::parameters param;
  lti::eCamFeatureMode brightness, saturation, hue;
  float brightness_value, saturation_value, hue_value;


  // try to read the configuration file
  std::ifstream in(confFile);

  if (in) {
    lti::lispStreamHandler lsh;
    lsh.use(in);
    param.read(lsh);
  }

  lti::v4l2 cam(param); // se cargan los paramentros del archivo "lsp" en la camara

  lti::image img;
  lti::channel x,last,y,s;

  lti::viewer2D::interaction action,lastAction;
  lti::viewer2D::parameters vPar;
  vPar.mappingType = lti::viewer2DPainter::Optimal;
  static lti::viewer2D view(vPar);
  static lti::viewer2D orig("Original");


  lti::ipoint pos;
  bool ok;

  if (!cam.apply(img)) { // si la camara no esta disponible, se imprime el error
    cout << "Error: " << cam.getStatusString() << endl;
    exit(EXIT_FAILURE);
  }
  last.assign(img.size(),0.0f);

  //se impriment los valores actuales de la camara para referencia del uauario

  cam.getBrightness(brightness,brightness_value);
  cam.getHue(hue,hue_value);
  cam.getSaturation(saturation,saturation_value);
  cout << "Valores actuales: " << endl;
  cout << "Brillo: " << brightness_value << endl;
  cout << "Saturacion: " << saturation_value << endl;
  cout << "Matiz: " << hue_value << endl;


  do {
    ok = cam.apply(img);

    orig.show(img); // se proyecta el frame actual


    orig.getLastAction(action,pos);
    if (action.action == lti::viewer2D::KeyPressed)  { // se detecta si se presiono una tecla
      switch (action.key) {
        case lti::viewer2D::EscKey: //si la tecla "Esc" es presionada se sale del programa
          ok = false;
          break;
        // si se detecta que se presiono un f/F se aumenta la saturacion
        case 68:
        case 100:
          saturation_value = saturation_value + 0.05;
          cam.setSaturation(saturation,saturation_value);
          cam.getSaturation(saturation,saturation_value);
          cout << "Saturacion actual: " << saturation_value << endl;
          break;
        // si se detecta que se presiono un d/D se disminuye la saturacion
        case 70:
        case 102:
          if(saturation_value<0.05){
            saturation_value = 0;
          }
          else{
            saturation_value = saturation_value - 0.05;
          }
          cam.setSaturation(saturation,saturation_value);
          cam.getSaturation(saturation,saturation_value);
          cout << "Saturacion actual: " << saturation_value << endl;
          break;
        // si se detecta que se presiono un a/A se aumenta el matiz
        case 65:
        case 97:
          hue_value = hue_value + 0.05;
          cam.setHue(hue,hue_value);
          cam.getHue(hue,hue_value);
          cout << "Matiz actual: " << hue_value << endl;
          break;
        // si se detecta que se presiono un s/S se disminuye el matiz
        case 83:
        case 115:
          hue_value = hue_value - 0.05;
          cam.setHue(hue,hue_value);
          cam.getHue(hue,hue_value);
          cout << "Matiz actual: " << hue_value << endl;
          break;
        // si se detecta que se presiono un b/B se aumenta el brillo
        case 66:
        case 98:
          brightness_value = brightness_value + 0.05;
          cam.setBrightness(brightness,brightness_value);
          cam.getBrightness(brightness,brightness_value);
          cout << "Brillo actual: " << brightness_value << endl;
          break;
        // si se detecta que se presiono un n/N se disminuye el brillo
        case 78:
        case 110:
          brightness_value = brightness_value - 0.05;
          cam.setBrightness(brightness,brightness_value);
          cam.getBrightness(brightness,brightness_value);
          cout << "Brillo actual: " << brightness_value << endl;
          break;
        // si se detecta que se presiono un h/H se imprime la ayda de usuario
        case 72:
        case 104:
          usage();
          break;
        default:
          break;
      }
    }
    lastAction = action;
  } while(ok);


  return EXIT_SUCCESS;
}
