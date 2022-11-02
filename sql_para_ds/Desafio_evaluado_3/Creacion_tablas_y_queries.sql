/*  Desafío Evaluado - Desafío 3
    Nombre: Thomas Peet, Braulio Aguila, Camilo Ramírez
    Generación: G47
    Profesores: José Terrazas - Sebastián Ulloa
    Fecha: 02-11-2022
*/

-- Por si acaso, para que no se llenen de nuevo las tablas.
DROP TABLE inventario_temp CASCADE;
DROP TABLE producto_1f CASCADE;
DROP TABLE compra_1f CASCADE;
DROP TABLE bodega_2f CASCADE;
DROP TABLE producto_2f CASCADE;
DROP TABLE compra_2f CASCADE;
DROP TABLE bodega_3f CASCADE;
DROP TABLE local_3f CASCADE;
DROP TABLE vendedor_3f CASCADE;
DROP TABLE cliente_3f CASCADE;
DROP TABLE producto_3f CASCADE;
DROP TABLE compra_3f CASCADE;
DROP TABLE bodega_has_producto_3f CASCADE;
DROP TABLE producto_has_bodega_3f CASCADE;


/*  
    Creación de la tabla temporal para alojar los csv que serán cargados a tablas con llaves foráneas.
*/

CREATE TABLE inventario_temp(
	codigo_producto INT,
	producto VARCHAR(50),
	local VARCHAR(100),
	precio INT,
	existencia VARCHAR(20),
	stock INT,
	ubicacion VARCHAR(100),
	numero_bodega INT,
	vendedor VARCHAR(50),
	rut_vendedor INT,
	numero_boleta INT,
	cantidad_vendida INT,
	rut_cliente INT,
	nombre_cliente VARCHAR(50)
);

/*  
    Llenado de datos desde CSV. Los archivos deben estar en la raíz y el script debe correrse desde una terminal ubicada en esta misma carpeta.
*/

\copy inventario_temp(codigo_producto, producto, local, precio, existencia, stock, ubicacion, numero_bodega,  vendedor, rut_vendedor, numero_boleta, cantidad_vendida, rut_cliente, nombre_cliente) FROM 'Apoyo desafio.csv' DELIMITER ',' CSV HEADER;

-- convierto los valores del csv de la variable existencia en valores booleanos.
ALTER TABLE inventario_temp ALTER existencia TYPE boolean USING CASE existencia WHEN 'TRUE' THEN true WHEN '1' THEN true WHEN 'Si' THEN true ELSE false END;

-- 1ERA FORMA NORMAL

-- El primer paso fue separar los productos de las compras. De esta forma evitamos tener grupos repetitivos de compras de productos iguales.

CREATE TABLE producto_1f(
	codigo_producto INT PRIMARY KEY, 
    producto VARCHAR(30), 
    precio INT, 
    existencia BOOLEAN,
    numero_bodega INT,
    local VARCHAR(100),
    ubicacion VARCHAR(100),
    stock INT
);

CREATE TABLE compra_1f(
	numero_boleta INT PRIMARY KEY,
    local VARCHAR(100),
    ubicacion VARCHAR(100),
    vendedor VARCHAR(50), 
    rut_vendedor INT, 
    nombre_cliente VARCHAR(50),
    rut_cliente INT, 
    cantidad_vendida INT,
    codigo_producto INT REFERENCES producto_1f(codigo_producto)
);

-- 2DA FORMA NORMAL

-- El segundo paso fue agregar la tabla bodega, que permite referenciar las compras y los productos a un lugar físico. Esto permite separar los atributos local y ubicacion de compra y productos.

CREATE TABLE bodega_2f(
	numero_bodega INT PRIMARY KEY,
	local VARCHAR(100),
	ubicacion VARCHAR(100)
);

CREATE TABLE producto_2f(
	codigo_producto INT PRIMARY KEY, 
    producto VARCHAR(30), 
    precio INT, 
    existencia BOOLEAN,
    stock INT,
    id_bodega INT REFERENCES bodega_2f(numero_bodega)
);

CREATE TABLE compra_2f(
	numero_boleta INT PRIMARY KEY,
    vendedor VARCHAR(50), 
    rut_vendedor INT, 
    nombre_cliente VARCHAR(50),
    rut_cliente INT, 
    cantidad_vendida INT,
    id_bodega INT REFERENCES bodega_2f(numero_bodega),
    codigo_producto INT REFERENCES producto_2f(codigo_producto)
);

-- 3RA FORMA NORMAL
-- El tercer paso fue agregar las tablas cliente, vendedor y local. La compra la hace un cliente y lo atiende un vendedor, pero estos vendedores y clientes pueden participar en más compras por lo que son entidades distintas y se relacionan por llaves foráneas. De la misma manera, los locales se separaron de bodega ya que puede haber más de una bodega en un local (casos bodegas 21 y 62 en local Las Rosas) pero solo un local por bodega.

-- Además de esto se agrego la tabla bodega_has_producto ya que era necesario relacionar cada producto a una bodega para poder almacenar el stock.

CREATE TABLE local_3f(
	local VARCHAR(100) PRIMARY KEY,
	ubicacion VARCHAR(100)
);

CREATE TABLE bodega_3f(
	numero_bodega INT PRIMARY KEY,
	id_local VARCHAR(100) REFERENCES local_3f(local)
);

CREATE TABLE vendedor_3f(
	rut_vendedor INT PRIMARY KEY,
	vendedor VARCHAR(50)
);

CREATE TABLE cliente_3f(
	rut_cliente INT PRIMARY KEY,
	nombre_cliente VARCHAR(50)
);


CREATE TABLE producto_3f(
	codigo_producto INT PRIMARY KEY, 
    producto VARCHAR(30), 
    precio INT, 
    existencia BOOLEAN
);

CREATE TABLE bodega_has_producto_3f(
	id_producto INT REFERENCES producto_3f(codigo_producto) NOT NULL,
	id_bodega INT REFERENCES bodega_3f(numero_bodega) NOT NULL,
	PRIMARY KEY (id_producto, id_bodega),
	stock INT
);

CREATE TABLE compra_3f(
	numero_boleta INT PRIMARY KEY,
    cantidad_vendida INT,
    id_vendedor INT REFERENCES vendedor_3f(rut_vendedor) NOT NULL, 
    id_cliente INT REFERENCES cliente_3f(rut_cliente) NOT NULL, 
    id_local VARCHAR(100) REFERENCES local_3f(local) NOT NULL,
    codigo_producto INT REFERENCES producto_3f(codigo_producto) NOT NULL
);


--  Por último se procede a llenar el modelo normalizado a través de la tabla en la que se cargaron los datos del csv.
INSERT INTO local_3f(local, ubicacion) SELECT DISTINCT ON (local) local, ubicacion FROM inventario_temp;
INSERT INTO bodega_3f(numero_bodega, id_local) SELECT DISTINCT ON (numero_bodega) numero_bodega, local FROM inventario_temp;
INSERT INTO vendedor_3f(rut_vendedor, vendedor) SELECT DISTINCT ON (rut_vendedor) rut_vendedor, vendedor FROM inventario_temp;
INSERT INTO cliente_3f(rut_cliente, nombre_cliente) SELECT DISTINCT ON (rut_cliente) rut_cliente, nombre_cliente FROM inventario_temp;
INSERT INTO producto_3f(codigo_producto, producto, precio, existencia) SELECT DISTINCT ON (codigo_producto) codigo_producto, producto, precio, existencia FROM inventario_temp;
INSERT INTO bodega_has_producto_3f(id_producto, id_bodega, stock) SELECT DISTINCT ON (codigo_producto, numero_bodega) codigo_producto, numero_bodega, stock FROM inventario_temp;
INSERT INTO compra_3f(numero_boleta, cantidad_vendida, id_vendedor, id_cliente, id_local, codigo_producto) SELECT DISTINCT ON (numero_boleta) numero_boleta, cantidad_vendida, rut_vendedor, rut_cliente, local, codigo_producto FROM inventario_temp;