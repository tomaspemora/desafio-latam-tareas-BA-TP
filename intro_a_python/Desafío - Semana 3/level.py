def choose_level(n_pregunta, p_level):
    
    # Construir lógica para escoger el nivel
    ##################################################
    
    if  1 <= n_pregunta <= p_level:
        level = 'basicas'
    elif n_pregunta <= p_level * 2: 
        level = 'intermedias'
    elif n_pregunta <= p_level * 3:
        level = 'avanzadas'
    else:
        print(f'Las variables n_pregunta : {n_pregunta} - p_level : {p_level} no representan un par válido para utilizar')
        level = 'no_valido'
    ##################################################
    
    return level

if __name__ == '__main__':
    # verificar resultados
    print(choose_level(2, 2)) # básicas
    print(choose_level(3, 2)) # intermedias
    print(choose_level(7, 3)) # avanzadas
    print(choose_level(4, 3)) # intermedias