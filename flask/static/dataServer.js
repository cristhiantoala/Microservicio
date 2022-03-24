let enviardatos;
let actualizardatos;
function enviar(){
  enviardatos = setInterval(obtenerData, 10000);
}
function actualizar(){
  actualizardatos = setInterval(actualizarData, 21000);
}
function obtenerData(){
  var hoy = new Date();
  var fecha = hoy.getFullYear() + '-' + ( hoy.getMonth() + 1 ) + '-' + hoy.getDate();
  var hora = hoy.getHours() + ':' + hoy.getMinutes() + ':' + hoy.getSeconds()+ ',' + hoy.getMilliseconds()+":";
  var fechaYHora = fecha + ' ' + hora;
  horaCiclo = fechaYHora+" "+"cierre_ciclo"
  var server_data = [
    {"Movimientos": movMouse},
    {"Teclas": pressTecla},
    {"Hora": horaCiclo}
  ];
  $.ajax({
    type: "POST",
    url: "/datos",
    data: JSON.stringify(server_data),
    contentType: "application/json",
    dataType: 'json',
    success: function(result) {
      console.log("Result:");
      console.log(result);
      animo=JSON.stringify(result.Emocion)
      cambiarEmocion(animo)
    } 
});
}

function actualizarData(){
  movMouse=[]
  pressTecla=[]
  horaCiclo=[]
}

enviar()
actualizar()