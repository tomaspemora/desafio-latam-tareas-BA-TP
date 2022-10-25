/*  Desafío Evaluado - Desafío 2
    Nombre: Thomas Peet, Braulio Aguila, Camilo Ramírez
    Generación: G47
    Profesores: José Terrazas - Sebastián Ulloa
    Fecha: 24-10-2022
*/

/*	1. Crear una base de datos mediante PostgreSQL basándonos en los artistas más
	escuchados en Spotify del 2018, para eso, es necesario que usted cree las siguientes
	tablas con sus respectivos datos:
		○ Artista.
			■ nombre_artista.
			■ fecha_de_nacimiento.
			■ nacionalidad.
		○ Canción.
			■ titulo_cancion.
			■ artista.
			■ album.
			■ numero_del_track.
		○ Albúm.
			■ titulo_album.
			■ artista.
			■ año.
*/

-- Por si acaso, para que no se llenen de nuevo las tablas.
DROP TABLE cancion CASCADE;
DROP TABLE album CASCADE;
DROP TABLE artista CASCADE;


CREATE TABLE artista(
	id_artista SERIAL PRIMARY KEY,
	nombre_artista VARCHAR(100),
	fecha_de_nacimiento DATE,
	nacionalidad VARCHAR(100)
);

CREATE TABLE album(
	id_album SERIAL PRIMARY KEY,
	titulo_album VARCHAR(100),
	anio INT,
	id_artista INT REFERENCES artista(id_artista)
);

CREATE TABLE cancion(
	id_cancion SERIAL PRIMARY KEY,
	titulo_cancion VARCHAR(100),
    numero_de_track INT,
    id_artista INT REFERENCES artista(id_artista),
    id_album INT REFERENCES album(id_album)
);

/*  
    Creación de tablas temporales para alojar los csv que serán cargados a tablas con llaves foráneas.
*/

CREATE TABLE album_temp(
	titulo_album VARCHAR(100),
	anio INT,
	artista VARCHAR(100)
);

CREATE TABLE cancion_temp(
	titulo_cancion VARCHAR(100),
    numero_de_track INT,
    artista VARCHAR(100),
    album VARCHAR(100)
);

/*  
    Llenado de datos desde CSV. Los archivos deben estar en la raíz y el script debe correrse desde una terminal ubicada en esta misma carpeta.
*/

\copy artista(nombre_artista, fecha_de_nacimiento, nacionalidad) FROM 'Artista.csv' DELIMITER ',' CSV HEADER;
\copy album_temp(titulo_album, artista, anio) FROM 'Album.csv' DELIMITER ',' CSV HEADER;
\copy cancion_temp(titulo_cancion,artista,album,numero_de_track) FROM 'Cancion.csv' DELIMITER ',' CSV HEADER;


/*  
    Llenado de la tabla album
*/
INSERT INTO album(titulo_album, anio, id_artista) 
SELECT album_temp.titulo_album, album_temp.anio, artista.id_artista 
FROM artista 
INNER JOIN album_temp ON artista.nombre_artista = album_temp.artista; 

/*  
    Llenado de la tabla cancion
*/
INSERT INTO cancion(titulo_cancion, numero_de_track, id_artista, id_album) 
SELECT cancion_temp.titulo_cancion, cancion_temp.numero_de_track, artista.id_artista, album.id_album
FROM cancion_temp
INNER JOIN album on album.titulo_album = cancion_temp.album 
INNER JOIN artista on artista.nombre_artista = cancion_temp.artista;

/*  
    Se dropean las tablas temporales
*/
DROP TABLE album_temp;
DROP TABLE cancion_temp;

/*	2. Ingrese los datos del archivo Artistas_populares_2018 a sus respectiva tabla y responda a las siguientes consultas:
		○ Canciones que salieron el año 2018.
		○ Albums y la nacionalidad de su artista.
		○ Número de track, canción, album, año de lanzamiento y artista donde las canciones deberán estar ordenadas por año 
		de lanzamiento del álbum, álbum	y artista correspondiente.
*/

-- P1
SELECT c.id_cancion, c.titulo_cancion, c.numero_de_track, a.titulo_album, a.anio
FROM  cancion c, album a 
where a.anio = 2018 and c.id_album = a.id_album;

-- P2
select a1.titulo_album, a2.nombre_artista, a2.nacionalidad
from album a1, artista a2
where a1.id_artista = a2.id_artista;

-- P3
select c.numero_de_track, c.titulo_cancion, a1.titulo_album, a1.anio, a2.nombre_artista
from cancion c, album a1, artista a2
where c.id_artista = a2.id_artista and c.id_album = a1.id_album
order by a1.anio, a1.titulo_album, a2.nombre_artista;
