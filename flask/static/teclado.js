function teclas(ev) {
    evento = ev || window.event;
    codigo = evento.keyCode;
    var hoy = new Date();
    var fecha = hoy.getFullYear() + '-' + ( hoy.getMonth() + 1 ) + '-' + hoy.getDate();
    var hora = hoy.getHours() + ':' + hoy.getMinutes() + ':' + hoy.getSeconds()+ ',' + hoy.getMilliseconds()+":";
    var fechaYHora = fecha + ' ' + hora;
    if (codigo==16){
        var tecla = fechaYHora+" "+"Key.shift"
        pressTecla.push(tecla)
        // document.getElementById("tecla").value = aux
    }
    if (codigo==32){
        var tecla = fechaYHora+" "+"Key.space"
        pressTecla.push(tecla)
        // document.getElementById("tecla").value = aux
    }
    if (codigo==8){
        var tecla = fechaYHora+" "+"Key.backspace"
        pressTecla.push(tecla)
        // document.getElementById("tecla").value = aux
    }
    if (codigo==13){
        var tecla = fechaYHora+" "+"Key.enter"
        pressTecla.push(tecla)
        // document.getElementById("tecla").value = aux
    }
    if (codigo==9){
        var tecla = fechaYHora+" "+"Key.tab"
        pressTecla.push(tecla)
        // document.getElementById("tecla").value = aux
    }
    if (codigo>=65 && codigo<=90){
        t=String.fromCharCode(codigo)
        t=t.toLowerCase()
        var tecla = fechaYHora+" "+t
        pressTecla.push(tecla)
        // document.getElementById("tecla").value = aux
    }
    }