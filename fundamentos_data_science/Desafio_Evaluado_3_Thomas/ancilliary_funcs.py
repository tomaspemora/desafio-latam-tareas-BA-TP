def puntaje_z(x):
    return (x - x.mean()) / x.std()


def puntaje_z_norm(x):
    return (x - x.mean()) / x.std()