if( navigator.userAgent.match(/Android/i)
|| navigator.userAgent.match(/webOS/i)
|| navigator.userAgent.match(/iPhone/i)
|| navigator.userAgent.match(/iPad/i)
|| navigator.userAgent.match(/iPod/i)
|| navigator.userAgent.match(/BlackBerry/i)
|| navigator.userAgent.match(/Windows Phone/i))
window.location.href = "https://google.com.ec";

// inicializamos un array de arrays con la preguntas del juego. 
var questionsopt = [
	[
		"En un microordenador tipo PC equipado con Microsoft Windows, el juego de teclas que permite iniciar el proceso de apagado del equipo es",
		"Ctrl + Alt + Supr",
		"Ctrl + SHIFT",
		"Ctrl + C",
		"Ninguno de los anteriores",
		0
	],
	[
		"¿Cuál de los siguientes prefijos define una mayor cantidad de bytes?",
		"Mega",
		"Kilo",
		"Tera",
		"Ninguno de los anteriores",
		2
	],
	[
		"¿En Internet el servicio que permite establecer una comunicación mediante texto escrito con otro interlocutor a través de una conexión con un servidor, se denomina?",
		"URL",
		"IRC o Chat",
		"FTP",
		"Ninguno de los anteriores",
		1
	],
	[
		"Un conjunto de 1024 Megabytes corresponde a:",
		"1 TERA",
		"1 GIGA",
		"100 BYTES",
		"Ninguno de los anteriores",
		1
	],
	[
		"¿Con cuantas papeleras de reciclaje cuenta MS Windows?",
		"UNA",
		"TRES",
		"UNA POR DISCO DURO",
		"Ninguno de los anteriores",
		2
	],
	[
		"¿Qué es un host?",
		"Un dominio",
		"Un puerto del ordenador mediante el cual puedes conectar varios dispositivos",
		"Un PC conectado a una red con una IP asignada",
		"Ninguno de los anteriores",
		2
	],

	[
		"¿Desde dónde se puede modificar las propiedades de pantalla en Windows?",
		"Desde el escritorio.",
		"Desde el Panel de Control",
		"Ambas son correctas",
		"Ninguno de los anteriores",
		2
	],
	[
		"¿Cúal de éstas es una herramienta de sistema propia de las que incorpora MS Windows?",
		"La calculadora",
		"El desfragmentador",
		"Excel",
		"Ninguno de los anteriores",
		1
	],
	[
		"El sistema operativo es:",
		"El software almacenado en el hardware.",
		"Un conjunto de programas que se encarga de controlar la computadora.",
		"La forma de almacenar datos.",
		"La unidad elemental de información.",
		1
	],
	[
		"¿Se puede recuperar un archivo borrado?",
		"No.",
		"Si, mediante una utilidad del sistema operativo.",
		"Si, después de vaciar la papelera de reciclaje.",
		"Si, utilizando la papelera de reciclaje.",
		3
	],
		[
		"¿Cuáles de las siguientes opciones son barras de menú específicas que tiene Excel?",
		"La barra de símbolos de Excel.",
		"La barra de herramientas estándar.",
		"La barra de panel de tareas.",
		"Ninguna de las anteriores.",
		1
	],
		[
		"Cuando realizamos dibujos y hacemos edición de imágenes en la computadora lo podemos hacer con algunos de estos programas, ¿Cuáles son?",
		"Paint.",
		"Fireworks.",
		"Photoshop.",
		"Todas las anteriores.",
		3
	],	
		[
		"Menciona ¿Qué son los elementos de entrada a un computador?",
		"Son programas en espera al ser ejecutados.",
		"Son los que permiten dar entrada al CPU.",
		"Son los que te permiten abrir internet.",
		"Google, Yahoo!, Gmail.",
		1
	],
		[
		"¿Dónde está situada la Barra de tareas?",
		"Está situada en la parte superior de la ventana.",
		"Está situada en la parte central de la ventana.",
		"Está situada en la parte inferior de la pantalla.",
		"Está situada en la parte superior de la pantalla.",
		2
	],	
		[
		"Indique ¿Cuál es la combinación de teclas que sirve para cerrar una ventana?",
		"ALT+F4",
		"CTL+ALT+Z",
		"CTL+Z",
		"ALT+SHIFT",
		0
	],	
		[
		"En Excel la fórmula que permite sumar los valores desde la celda A1 hasta la celda A33 es",
		"=(A1:A33)",
		"=SUMA(A1:A33)",
		"SUMA(A1:A33)",
		"=SUMA(A1+A33)",
		1
	],	
		[
		"Cuando abrimos Word ¿Qué menú es el que nos aparece por defecto?",
		"Archivo.",
		"Inicio.",
		"Herramientas.",
		"Edición.",
		1
	],	
		[
		"Para poder mover los márgenes de Izquierda-Derecha ¿Qué herramientas ocupamos?",
		"Reglas.",
		"Regla vertical.",
		"Sangría.",
		"Espaciado.",
		2
	],	
		[
		"¿Cómo se les llaman a los atributos que tiene un documento?",
		"Estilo.",
		"Formato.",
		"Fuentes.",
		"Tema.",
		0
	],	
		[
		"¿Con qué herramientas podemos hacer que una palabra sobresalga del demás texto?",
		"Resaltado o negrita.",
		"Sombreado.",
		"Cambio de color.",
		"Borde.",
		0
	],	
		[
		"Las opciones de Negritas, Cursivas, Subrayado, Tachado, Tamaño, etc., ¿Cómo se la conoce a todas esas herramientas?",
		"Herramientas de Edición.",
		"Herramientas de Fuente.",
		"Herramienta de Formato.",
		"Herramienta de Buscar.",
		1
	],	
		[
		"Teniendo 4 párrafos queremos alinear el texto, centrarlo, moverlo, etc., ¿Qué herramienta nos ayudaría?",
		"Herramientas de Inicio.",
		"Herramientas de Fuente.",
		"Herramienta de Buscar.",
		"Herramienta de Párrafo.",
		3
	],	
		[
		"¿Qué combinación de teclas me permite copiar un documento?",
		"CTRL+B",
		"CTRL+SHIFT",
		"CTRL+C",
		"CTRL+T",
		2
	],	
		[
		"¿Qué combinación de teclas me permite pegar el contenido copiado?",
		"CTRL+V",
		"CTRL+P",
		"CTRL+C",
		"CTRL+Z",
		0
	],	
		[
		"Qué botón me despliega todos los programas que están instalados en la maquina",
		"Inicio.",
		"Todos los programas.",
		"Panel de tareas.",
		"Restaurar.",
		0
	],	
		[
		"Para poder crear espacios más amplios entre líneas y líneas ¿Qué herramienta ocupamos?",
		"Marcas.",
		"Espaciado.",
		"Interlineado.",
		"Sangrías.",
		2
	],	
		[
		"Todos los equipos para su correcto funcionamiento deben tener",
		"Procesador de textos.",
		"Un sistema operativo.",
		"Una impresora conectada.",
		"Navegador.",
		1
	],	
		[
		"¿Qué extensión tiene los documentos que se realizan en Word?",
		".PDF",
		".TXT",
		".JPG",
		".DOC/DOCX",
		3
	],	
		[
		"La unidad donde se insertan los CDs se denomina",
		"Disquetera.",
		"Impresora.",
		"Lector CD/DVD.",
		"Lector USB.",
		2
	],	
		[
		"¿Qué terminación tiene aquellos archivos realizados con el Excel?",
		".EXC",
		".XLT",
		".XLS",
		".XLC",
		2
	],	
		[
		"¿Qué es Excel?",
		"Una página Web.",
		"Un programa para Linux.",
		"Un programa para realizar hojas de cálculos.",
		"Un fichero de texto.",
		2
	],	
		[
		"Qué teclas debes utilizar para seleccionar todo en Windows",
		"CTRL+B",
		"CTRL+A",
		"CTRL+E",
		"CTRL+T",
		2
	],	
		[
		"¿Qué es un hipervínculo?",
		"Un lenguaje de programación.",
		"Un enlace.",
		"Un compilador.",
		"Un editor de fotos.",
		1
	],	
		[
		"¿Cómo se conoce al sistema de numeración que solo utiliza dos dígitos, 0 y 1?",
		"Decimal.",
		"Hexadecimal.",
		"Binario.",
		"Octal.",
		2
	],	
		[
		"El termino _________ se refiere a componentes físicos de la computadora, como el monitor, el teclado, la memoria y el disco duro.",
		"Firmware.",
		"Salida.",
		"Hardware.",
		"Software.",
		2
	],	
		[
		"Los computadores constan de 2 componentes básicos los cuales son:",
		"Pantalla y monitor.",
		"Hardware y Software.",
		"Internet y CPU.",
		"Hard drive pendrive.",
		1
	],
		[
		"¿Cómo se le llama a la hoja que usamos en Word?",
		"Hoja de Cálculo.",
		"Página.",
		"Diapositiva.",
		"Cuadro.",
		1
	],	
		[
		"Es un editor de presentaciones donde se integran textos, gráficos, imágenes, plantillas, animaciones, efectos de sonido y video",
		"Word.",
		"Access.",
		"Excel.",
		"Power Point.",
		3
	],	
		[
		"¿Qué nombre recibe la hoja usada en Power Point?",
		"Hoja de Cálculo.",
		"Página.",
		"Diapositiva.",
		"Cuadro.",
		2
	],	
		[
		"En Power Point ¿Cómo se denomina el archivo creado?",
		"Hoja de Cálculo.",
		"Documento.",
		"Presentación.",
		"Libro de trabajo.",
		2
	],	
		[
		"¿Cómo se llama a cada uno de los espacios que tenemos en Excel para escribir contenido en ellas?",
		"Celda.",
		"Rectángulo.",
		"Cuadrado.",
		"Espacio.",
		0
	],	
		[
		"Es un conjunto de redes de comunicación con servicios como el envío de correo electrónico, la transmisión de archivos, las conversaciones en línea, la mensajería instantánea, la transmisión de contenido y comunicación multimedia.",
		"Red social.",
		"Word.",
		"Correo electrónico.",
		"Internet.",
		3
	],	
		[
		"Es una estructura social compuesta por un conjunto de individuos u organizaciones que están relacionados en lo profesional, amistad, parentesco.",
		"Red social.",
		"Word.",
		"Correo electrónico.",
		"Internet.",
		0
	],	
		[
		"Es un servicio de red que permite a los usuarios enviar y recibir mensajes (también denominados mensajes electrónicos o cartas digitales) mediante redes de comunicación electrónica.",
		"Red social.",
		"Word.",
		"Correo electrónico.",
		"Internet.",
		2
	],	
		[
		"En Power Point ¿Qué pestaña nos permite dar un cambio de movimiento a toda la diapositiva?",
		"Transiciones.",
		"Diseño.",
		"Inicio.",
		"Revisar.",
		0
	],	
		[
		"En Power Point ¿Qué pestaña nos permite dar movimiento solo a texto e imágenes?",
		"Revisar.",
		"Animaciones.",
		"Inicio.",
		"Diseño.",
		1
	],	
		[
		"En Power Point ¿Qué pestaña nos permite agregar imágenes, tablas y gráficos?",
		"Revisar.",
		"Insertar.",
		"Inicio.",
		"Diseño.",
		1
	],	
		[
		"El archivo que creamos en Excel es:",
		"Hoja de Cálculo.",
		"Documento.",
		"Presentación.",
		"Libro de trabajo.",
		3
	],	
		[
		"En Word ¿Cómo se le llama al tipo de letra que utilizamos?",
		"Fuente.",
		"Letra.",
		"Palabras.",
		"Texto.",
		0
	],	
		[
		"¿Qué podemos crear en Word?",
		"Presupuestos.",
		"Presentaciones.",
		"Juegos.",
		"Documentos.",
		3
	],	
		[
		"El área de trabajo en Excel está conformado por:",
		"Columnas, Filas y Celdas.",
		"Columnas, Páginas y Cuadros de Texto.",
		"Columnas, Párrafos y Celdas.",
		"Cuadros, Filas y Celdas.",
		0
	],	
		[
		"¿Cuál de las siguientes opciones corresponde a tipos de Sistemas Operativos Existentes?",
		"Office, Word, Excel.",
		"Mother Board, Mouse, Teclado.",
		"Windows, Mac Os, Linux.",
		"Internet, Servidor, Conexión FTP.",
		2
	],	
		[
		"¿Qué programa utilizamos para escribir cartas?",
		"Word.",
		"Excel.",
		"Chrome.",
		"Power Point.",
		0
	],	
		[
		"¿Cuál de los siguientes dispositivos es apropiado para almacenar información?",
		"Memoria RAM.",
		"Disco Duro.",
		"Memoria Secundaria.",
		"Condensador.",
		1
	],	
		[
		"¿Cuál de las siguientes extensiones se crea un archivo comprimido en el sistema operativo Windows?",
		".RAR",
		".ZIP",
		".HQX",
		"Todas las anteriores.",
		3
	],	
		[
		"Los programas usados para tener acceso a páginas web y desplegarlas se llaman:",
		"Sitios Web.",
		"Ventanas Web.",
		"Documentos Web.",
		"Exploradores Web.",
		3
	],	
		[
		"Las tres primeras pestañas de la ventana de Power Point son:",
		"Inicio, Insertar y Diseño.",
		"Inicio, Animación y Presentación de diapositiva.",
		"Inicio, Animación y Revisar.",
		"Inicio, Diseño, Presentación de diapositiva.",
		2
	],	
		[
		"El error #¡VALOR! en una formula en Excel significa",
		"Excel no reconoce el texto de la fórmula, por lo que no puede realizar la operación matemática.",
		"Especifica una intersección de dos áreas que no se interceptan.",
		"Función o fórmula no disponible en Excel.",
		"No encuentra el valor de resultado de la formula.",
		0
	]	
];
var arrayAleatorio= [];
function randonQuestion(){
	var cantidadNumeros = 8;
	while(arrayAleatorio.length < cantidadNumeros ){
		var numeroAleatorio = Math.ceil(Math.random()*(questionsopt.length-1));
		var existe = false;
		for(var i=0;i<arrayAleatorio.length;i++){
			if(arrayAleatorio [i] == numeroAleatorio){
				existe = true;
				break;
			}
		}
		if(!existe){
			arrayAleatorio[arrayAleatorio.length] = numeroAleatorio;
		}

	}
}
randonQuestion()
var questions = [
	[
	questionsopt[arrayAleatorio[0]][0],
	questionsopt[arrayAleatorio[0]][1],
	questionsopt[arrayAleatorio[0]][2],
	questionsopt[arrayAleatorio[0]][3],
	questionsopt[arrayAleatorio[0]][4],
	questionsopt[arrayAleatorio[0]][5]
	],
	[
		questionsopt[arrayAleatorio[1]][0],
		questionsopt[arrayAleatorio[1]][1],
		questionsopt[arrayAleatorio[1]][2],
		questionsopt[arrayAleatorio[1]][3],
		questionsopt[arrayAleatorio[1]][4],
		questionsopt[arrayAleatorio[1]][5]
	],
	[
		questionsopt[arrayAleatorio[2]][0],
		questionsopt[arrayAleatorio[2]][1],
		questionsopt[arrayAleatorio[2]][2],
		questionsopt[arrayAleatorio[2]][3],
		questionsopt[arrayAleatorio[2]][4],
		questionsopt[arrayAleatorio[2]][5]
	],
	[
		questionsopt[arrayAleatorio[3]][0],
		questionsopt[arrayAleatorio[3]][1],
		questionsopt[arrayAleatorio[3]][2],
		questionsopt[arrayAleatorio[3]][3],
		questionsopt[arrayAleatorio[3]][4],
		questionsopt[arrayAleatorio[3]][5]
	],
	[
		questionsopt[arrayAleatorio[4]][0],
		questionsopt[arrayAleatorio[4]][1],
		questionsopt[arrayAleatorio[4]][2],
		questionsopt[arrayAleatorio[4]][3],
		questionsopt[arrayAleatorio[4]][4],
		questionsopt[arrayAleatorio[4]][5]
	],
	[
		questionsopt[arrayAleatorio[5]][0],
		questionsopt[arrayAleatorio[5]][1],
		questionsopt[arrayAleatorio[5]][2],
		questionsopt[arrayAleatorio[5]][3],
		questionsopt[arrayAleatorio[5]][4],
		questionsopt[arrayAleatorio[5]][5]
	],
	[
		questionsopt[arrayAleatorio[6]][0],
		questionsopt[arrayAleatorio[6]][1],
		questionsopt[arrayAleatorio[6]][2],
		questionsopt[arrayAleatorio[6]][3],
		questionsopt[arrayAleatorio[6]][4],
		questionsopt[arrayAleatorio[6]][5]
	],
	[
		questionsopt[arrayAleatorio[7]][0],
		questionsopt[arrayAleatorio[7]][1],
		questionsopt[arrayAleatorio[7]][2],
		questionsopt[arrayAleatorio[7]][3],
		questionsopt[arrayAleatorio[7]][4],
		questionsopt[arrayAleatorio[7]][5]
	],
	[
		"¿Cuál es la diferencia entre Hardware y Software?",
		1
	],
	[
		"¿Qué tal se sintió durante el test?",
		1
	],
	[
		"De los temas mencionados en el test ¿sobre cuáles tienes más conocimientos?",
		1
	]
];

// Aquí utilizamos UnderscoreJS para generar un template de pregunta.
var questionTemplate = _.template(" \
	<div class='card question'><span class='question'><%= question %></span> \
      <ul class='options'> \
        <li> \
          <input type='radio' name='question[<%= index %>]' title='hola' value='0' id='q<%= index %>o1'> \
          <label for='q<%= index %>o1'><%= a %></label> \
        </li> \
        <li> \
          <input type='radio' name='question[<%= index %>]' value='1' id='q<%= index %>o2'> \
          <label for='q<%= index %>o2'><%= b %></label> \
        </li> \
        <li> \
          <input type='radio' name='question[<%= index %>]' value='2' id='q<%= index %>o3'> \
          <label for='q<%= index %>o3'><%= c %></label> \
        </li> \
        <li> \
          <input type='radio' name='question[<%= index %>]' value='3' id='q<%= index %>o4'> \
          <label for='q<%= index %>o4'><%= d %></label> \
        </li> \
      </ul> \
    </div> \
    ");
	var questionTemplatetxt1 = _.template(" \
	<div class='card question'><span class='question'><%= question %></span> \
	<ul class='optionss'> \
	<li> \
          <textarea name='txtRes' id='txt_Res1'></textarea> \
		  </li> \
		<ul class='options'> \
        <li id='texto'> \
		<span id='hover1' style='position:relative;font:bold 16px sans-serif'></span> \
          <input type='radio' name='question[<%= index %>]' value='1' id='q<%= index %>o2' title='hola'> \
          <label id= 'Q1' for='q<%= index %>o2'><%= b %>Aceptar</label> \
        </li> \
		Caracteres: <span id='numerocaracteres1'>0</span> \
      </ul> \
    </div> \
    ");
	var questionTemplatetxt2 = _.template(" \
	<div class='card question'><span class='question'><%= question %></span> \
	<ul class='optionss'> \
	<li> \
          <textarea name='txtRes' id='txt_Res2'></textarea> \
		  </li> \
		<ul class='options'> \
        <li id='texto2'> \
		<span id='hover2' style='position:relative;font:bold 16px sans-serif'></span> \
          <input type='radio' name='question[<%= index %>]' value='1' id='q<%= index %>o2' > \
          <label id= 'Q2' for='q<%= index %>o2'><%= b %>Aceptar</label> \
        </li> \
		Caracteres: <span id='numerocaracteres2'>0</span> \
      </ul> \
    </div> \
    ");
	var questionTemplatetxt3 = _.template(" \
	<div class='card question'><span class='question'><%= question %></span> \
	<ul class='optionss'> \
	<li> \
          <textarea name='txtRes' id='txt_Res3'></textarea> \
		  </li> \
		<ul class='options'> \
        <li id='texto3'> \
		<span id='hover3' style='position:relative;font:bold 16px sans-serif'></span> \
          <input type='radio' name='question[<%= index %>]' value='1' id='q<%= index %>o2' > \
          <label id= 'Q3' for='q<%= index %>o2'><%= b %>Aceptar</label> \
        </li> \
		Caracteres: <span id='numerocaracteres3'>0</span> \
      </ul> \
    </div> \
    ");


// Definimos las variables de estado del juego y los valores iniciales (como el tiempo de respuesta de cada pregunta).
var points,
	pointsPerQuestion,
	currentQuestion,
	questionTimer,
	timeForQuestion = 30, // segundos
	timeLeftForQuestion; 

// Manipulacion de elementos con JQuery.
$(function() {

	// Uso de jQuery para escuchar el evento click del botón de Comenzar y Volver a jugar.
	$('button.start').click(start);
	$('.play_again button').click(restart);

	// La función restart inicializa los valores de las variables de estado del juego.
	function restart() {
		points = 0;
		pointsPerQuestion = 8;
		currentQuestion = 0;
		timeLeftForQuestion = timeForQuestion;
	// Se oculta la pantalla de finalizar y un mensaje que dice "Se acabó el tiempo".
		$('.finish.card').hide();
		$('div.start').show();
		$('.times_up').hide();

		generateCards();
		//updateTime();
	}

	//  La función start se ejecuta cuando el jugador hace click en comenzar.
	function start() {
		$('div.start').fadeOut(200, function() {
			moveToNextQuestion();
		});
	}

	
	

	// Esta es una de las funciones clave del juego, encargada de generar las preguntas. 
	function generateCards() {
		$('.questions').html('');
		for (var i = 0; i < questions.length; i++) {
			if (i==8){
				var q = questions[i];
				var html = questionTemplatetxt1({
					question: q[0],
					index: i,
					a: q[1],
					b: q[2],
				});
			}
			else if (i==9){
				var q = questions[i];
				var html = questionTemplatetxt2({
					question: q[0],
					index: i,
					a: q[1],
					b: q[2],
				});
			}
			else if (i==10){
				var q = questions[i];
				var html = questionTemplatetxt3({
					question: q[0],
					index: i,
					a: q[1],
					b: q[2],
				});
			}
			else{
				var q = questions[i];
				var html = questionTemplate({
					question: q[0],
					index: i,
					a: q[1],
					b: q[2],
					c: q[3],
					d: q[4]
				});
			}
			$('.questions').append(html);
		};

                // Indicamos que nos interesa el evento change de los inputs dentro de los elementos con clase question y card (cada una de las preguntas).
		$('.question.card input').change(optionSelected);
		/*$('#texto').mouseover(optionFocus)
		$('#texto2').mouseover(optionFocus)
		$('#texto3').mouseover(optionFocus)*/
		$("#texto").hover(hoveron,hoveroff)
		$("#texto2").hover(hoveron,hoveroff)
		$("#texto3").hover(hoveron,hoveroff)
	}

	// Esta función cambia el estado del juego para pasar de una pregunta a la siguiente.
	function moveToNextQuestion() {
		currentQuestion += 1;
		if (currentQuestion > 1) {
			$('.question.card:nth-child(' + (currentQuestion-1) + ')').hide();
			
		}


		// Se muestra la siguiente pregunta.
		showQuestionCardAtIndex(currentQuestion);

	}

	// Mostramos la tarjeta de pregunta correspondiente al índice que la función recibe por parámetro.
	function showQuestionCardAtIndex(index) { // staring at 1
		
		var $card = $('.question.card:nth-child(' + index + ')').show();
	}

	// Actualiza los puntos en pantalla.
	function updatePoints() {
		$('.points span.points').html(points + ' puntos');
	}


	// Esta función se ejecuta cuando el jugador escoge una respuesta.
	function optionSelected() {
		var aux = currentQuestion-1;
		var aux2 = questions[aux].length;
		var selected = parseInt(this.value);
		var correct = questions[aux][aux2-1];
		
		if (selected == correct) {
			if(aux!=8 && aux!=9 && aux!=10){
				points += pointsPerQuestion;
				updatePoints();
				correctAnimation();
			}
		} else if(aux!=8 && aux!=9 && aux!=10) {

			wrongAnimation();
		}

		if (currentQuestion == questions.length) {
			clearTimeout(questionTimer);
			return finish();
		}
		moveToNextQuestion();
	}

	function contador(idtextarea, idcontador){

		function update_contador(idtextarea, idcontador){
	
			var contador = $(idcontador);
			var ta = $(idtextarea);
			
			contador.html(ta.val().replace(/\s+/gi, '').replace(/[ ]+/g,'').length);
	
		}
	
		$(idtextarea).keyup(function(){
			update_contador(idtextarea, idcontador);
		});
		
		$(idtextarea).change(function(){
			update_contador(idtextarea, idcontador);
		});
			
	}
	
	$( document ).ready(function() {
	
		contador("#txt_Res1","#numerocaracteres1");
		contador("#txt_Res2","#numerocaracteres2");
		contador("#txt_Res3","#numerocaracteres3");
	});
	
	function hoveron() {
		var CQ = currentQuestion-1;
			if (CQ==8)
			{
				var txt=document.getElementById("txt_Res1").value
				//Contar palabras del texto.
				var wordCount = txt.trim().replace(/\s+/gi, '').replace(/[ ]+/g,'').length;
				if (wordCount<50)
				{
					document.getElementById("hover1").innerHTML="¡Debe insertar al menos 50 caracteres!";
					o=document.getElementById("hover1");
					o.style.left=2+"px";
					o.style.top=-10+"px";
					$("#q8o2").attr("disabled","disabled")
					$("#Q1").attr("title","Debe ingresar al menos 50 caracteres")
				}
			}
			if (CQ==9)
			{
				
				var txt=document.getElementById("txt_Res2").value
				//Contar palabras del texto.
				var wordCount = txt.trim().replace(/\s+/gi, '').replace(/[ ]+/g,'').length;
				if (wordCount<50)
				{
					document.getElementById("hover2").innerHTML="¡Debe insertar al menos 50 caracteres!";
					o=document.getElementById("hover2");
					o.style.left=2+"px";
					o.style.top=-10+"px";
					$("#q9o2").attr("disabled","disabled")
					$("#Q2").attr("title","Debe ingresar al menos 50 caracteres")
				}
			}
			if (CQ==10)
			{
				var txt=document.getElementById("txt_Res3").value
				//Contar palabras del texto.
				var wordCount = txt.trim().replace(/\s+/gi, '').replace(/[ ]+/g,'').length;
				if (wordCount<50)
				{
					document.getElementById("hover3").innerHTML="¡Debe insertar al menos 50 caracteres!";
					o=document.getElementById("hover3");
					o.style.left=2+"px";
					o.style.top=-10+"px";
					$("#q10o2").attr("disabled","disabled")
					$("#Q3").attr("title","Debe ingresar al menos 50 caracteres")
				}
			}
		}
		function hoveroff(){
			var CQ = currentQuestion-1;
			if (CQ==8)
			{
				document.getElementById("hover1").innerHTML="";
				o=document.getElementById("hover1");
				$("#q8o2").removeAttr("disabled")
			}
				if (CQ==9)
			{
				document.getElementById("hover2").innerHTML="";
				o=document.getElementById("hover2");
				$("#q9o2").removeAttr("disabled")
			}
				if (CQ==10)
			{
				document.getElementById("hover3").innerHTML="";
				o=document.getElementById("hover3");
				$("#q10o2").removeAttr("disabled")
			}
		}
	// Animación de respuesta correcta e incorrecta.
	function correctAnimation() {
		animatePoints('right');
	}

	// Animación de respuesta correcta e incorrecta.
	function wrongAnimation() {
		animatePoints('wrong');
	}

	// Esta función anima el puntaje en pantalla.
	function animatePoints(cls) {
		$('header .points').addClass('animate ' + cls);
		var body=$("body")
		body.addClass('animate ' + cls)
		setTimeout(function() {
			var body=$("body")
			body.removeClass('animate ' + cls)
			$('header .points').removeClass('animate ' + cls);
		}, 500);
	}

	// Cuando el juego termina, esta función es ejecutada.
	function finish() {
		var acertadas = points / 8
		var incorrectas = 8-acertadas
		if (timeLeftForQuestion == 0) {
			$('.times_up').show();
		}
		//$('p.final_points').html(points + ' puntos');
		$('p.acertadas').html(acertadas + ' aciertos');
		$('p.incorrectas').html(incorrectas + ' fallos');
		$('.question.card:visible').hide();
		$('.finish.card').show();
	}

	// 24
	restart();

});