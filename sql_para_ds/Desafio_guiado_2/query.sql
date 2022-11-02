select c.*
from artista a2, cancion c
where a2.fecha_de_nacimiento >= '1992-01-01' 
and c.numero_de_track = 4 
and a2.id_artista  = c.id_artista  
and a2.nacionalidad = 'Estadounidense'
LIMIT 1