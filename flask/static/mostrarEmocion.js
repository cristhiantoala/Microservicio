function cambiarEmocion(emocionA) {
  emocionA=emocionA.replace(/[\\'\""]/g, '');
  h=document.querySelector("header")
  if (emocionA=="Positivo"){
    var header=$(".title")
			header.css("background-color","#3EC642")
  }
  if (emocionA=="Negativo"){
    var header=$(".title")
    header.css("background-color","#FF0000")
  }
  if (emocionA=="Neutro"){
    var header=$(".title")
    header.css("background-color","#B9BDB3")
  }
  var hoy = new Date();
  var hora = hoy.getHours() + ':' + hoy.getMinutes()+ ':' + hoy.getSeconds();
  horaEmocion="Emocion: "+emocionA+"<br>"+"Hora: "+hora
  document.getElementById("emocion").innerHTML=horaEmocion
}
