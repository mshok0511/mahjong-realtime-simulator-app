document.addEventListener('DOMContentLoaded', (event) => {
    const socket = io();
    const toggleButton = document.getElementById("toggleButton");
    var capturing = false;

    toggleButton.addEventListener("click",function(){
        if (capturing) {
            //ストップ処理
            socket.emit('stop_capture');
            toggleButton.innerText = 'Start Capture';
        } else {
            //スタート処理
            socket.emit('start_capture');
            toggleButton.innerText = 'Stop Capture';
        }
        capturing = !capturing;
    });

    document.getElementById("calc").addEventListener("click",function(){
        socket.emit('calc');
    });

    socket.on('new_image', function(data) {
        if (capturing) {
            let img = document.getElementById("screenshot");
            img.src = data.img_path
        }
    });

    socket.on('error', function(data) {
        const errorBox = document.getElementById("error");
        errorBox.innerHTML = '<div class="error">' + data.error + '</div>';
    });
    socket.on('error_calc', function(data) {
        const errorcBox = document.getElementById("error_calc");
        errorcBox.innerHTML = data.error_calc;
    });

    socket.on('tehaiList', function(data) {
        const tehaiListBox = document.getElementById("tehaiList");
        tehaiListBox.innerHTML = "";

        data.tehaiList.forEach(item => {
            const listItem = document.createElement('li');
            listItem.textContent = item;
            tehaiListBox.appendChild(listItem);
        });
    });

    socket.on('tsumohai', function(data) {
        const tsumohaiBox = document.getElementById("tsumohai");
        tsumohaiBox.innerHTML = data.tsumohai;
    });

    socket.on('doraList', function(data) {
        const doraListBox = document.getElementById("doraList");
        doraListBox.innerHTML = "";

        data.doraList.forEach(item => {
            const listItem = document.createElement('li');
            listItem.textContent = item;
            doraListBox.appendChild(listItem);
        });
    });

    const radios = document.querySelectorAll('input[type="radio"][name="option"]');
    radios.forEach((radio) => {
        radio.addEventListener('change', () => {
            const selectedValue = parseInt(document.querySelector('input[type="radio"][name="option"]:checked').value, 10);
            socket.emit('radio_change', { value: selectedValue });
        });
    });

    // デフォルト選択された状態をサーバーに送信する
    const selectedValue = parseInt(document.querySelector('input[type="radio"][name="option"]:checked').value, 10);
    socket.emit('radio_change', { value: selectedValue });

    socket.on('result', function(data) {
        const resultListBox = document.getElementById("response");
        resultListBox.innerHTML = "";
    
        data.result.forEach(item => {
            const listItem = document.createElement('li');
            listItem.textContent = item;
            resultListBox.appendChild(listItem);
        });
    });

    const checkboxes = document.querySelectorAll('input[type="checkbox"][name="option"]');
    checkboxes.forEach((checkbox) => {
        checkbox.addEventListener('change', () => {
            const selectedValues = Array.from(document.querySelectorAll('input[type="checkbox"][name="option"]:checked')).map(cb => parseInt(cb.value, 10));
            socket.emit('checkbox_change', { values: selectedValues });
        });
    });

    const numberInput = document.getElementById('tentacles');

    const defaultValue = parseInt(numberInput.value, 10);
    socket.emit('number_input', { value: defaultValue });

    numberInput.addEventListener('input', () => {
        const value = parseInt(numberInput.value, 10);
        socket.emit('number_input', { value: value });
    });


    const inputField = document.getElementById('tentacles');
    const decrementButton = document.getElementById('decrement');
    const incrementButton = document.getElementById('increment');

    // マイナスボタンをクリックしたときの処理
    decrementButton.addEventListener('click', () => {
        const currentValue = parseInt(inputField.value);
        if (currentValue > parseInt(inputField.min)) {
            inputField.value = currentValue - 1;
        }
    });

    // プラスボタンをクリックしたときの処理
    incrementButton.addEventListener('click', () => {
        const currentValue = parseInt(inputField.value);
        if (currentValue < parseInt(inputField.max)) {
            inputField.value = currentValue + 1;
        }
    });
});