const {PythonShell} = require('python-shell');
const electron = require('electron');
const app = electron.app;
const BrowserWindow = electron.BrowserWindow;
let mainWindow;

app.on('ready', function() {
    PythonShell.run('./app.py');
    const openWindow = function() {
        mainWindow = new BrowserWindow({width: 800, height: 600 });
        mainWindow.setMenuBarVisibility(false); 
        mainWindow.loadURL('http://localhost:5000');
    };
    openWindow();
});