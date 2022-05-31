import preguntas as p
import random
from shuffle import shuffle_alt

# Opciones dadas para escoger.
###############################################
opciones = {'basicas': [1,2,3],
            'intermedias': [1,2,3],
            'avanzadas': [1,2,3]}
###############################################

def choose_q(dificultad):
    #escoger preguntas por dificultad
    if dificultad == 'basicas':
        preguntas = p.preguntas_basicas
    elif dificultad == 'intermedias':
        preguntas = p.preguntas_intermedias
    elif dificultad == 'avanzadas':
        preguntas = p.preguntas_avanzadas

    # usar opciones desde ambiente global
    global opciones
    
    # escoger una pregunta
    
    opcion = opciones[dificultad]
    n_elegido = random.choice(opcion)
    
    # eliminarla del ambiente global para no escogerla de nuevo
    opcion.remove(n_elegido)
    
    # escoger enunciado y alternativas mezcladas
    pregunta = preguntas[f'pregunta_{n_elegido}']
    alternativas = shuffle_alt(pregunta)
        
    return pregunta['enunciado'], alternativas

if __name__ == '__main__':
    # si ejecuto el programa, las preguntas cambian de orden, pero nunca debieran repetirse
    pregunta, alternativas = choose_q('basicas')
    print(f'El enunciado es: {pregunta}')
    print(f'Las alternativas son: {alternativas}')
    
    pregunta, alternativas = choose_q('basicas')
    print(f'El enunciado es: {pregunta}')
    print(f'Las alternativas son: {alternativas}')
    
    pregunta, alternativas = choose_q('basicas')
    print(f'El enunciado es: {pregunta}')
    print(f'Las alternativas son: {alternativas}')