import preguntas as p
from shuffle import shuffle_alt

def verificar(alternativas, eleccion):
    #devuelve el índice de elección dada
    eleccion = ['a', 'b', 'c', 'd'].index(eleccion)

    # generar lógica para determinar respuestas correctas
    ##########################################################################################
    if alternativas[eleccion][1] == 1:
        print("Respuecta correcta")
    else:
        print("Respuesta incorrecta")
        
    correcto = alternativas[eleccion][1] == 1
    ##########################################################################################
    return correcto



if __name__ == '__main__':
    from print_preguntas import print_pregunta
    
    # Siempre que se escoja la alternativa con alt_2 estará correcta, e incorrecta en cualquier otro caso
    pregunta = p.pool_preguntas['basicas']['pregunta_2']
    print_pregunta(pregunta['enunciado'], pregunta)
    respuesta = input('Escoja la alternativa correcta:\n> ').lower()
    verificar(pregunta['alternativas'], respuesta)






