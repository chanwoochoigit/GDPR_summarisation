var express = require('express');
var bodyParser = require('body-parser');
var app = express();
var port = 8890;

app.set('port', process.env.PORT || 80);
app.use(bodyParser.urlencoded({extended:true, limit: '10mb'}));
//app.use(bodyParser.json({limit: '10mb'}));
app.use(bodyParser.json());
app.use(function (req, res, next) { //1
  res.header('Access-Control-Allow-Origin', '*');
  res.header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE');
  res.header('Access-Control-Allow-Headers', 'content-type');
  next();
});

const {PythonShell} = require('python-shell');

var options = {
    mode: 'text',
    pythonPath: '',
    pythonOptions: ['-u'],
    scriptPath: '',
    args: []
};

app.post('/getAPI', function(req, res){
    var data = req.body;
    console.log('data - ' + JSON.stringify(data));
	options.args = [];
    options.args.push(JSON.stringify(data));
    PythonShell.run('/root/worth_reading_finder/main.py', options, function (err, results) {
        var returnMsg = {
            errCode : '0000',
            errMsg : 'Successful'
        }
        if (err) {
            console.log(err);
            returnMsg.errCode = '1111';
            returnMsg.errMsg = 'Error';
            returnMsg.data = err;
            res.json(returnMsg);
            res.end();
        };
       // console.log('results: ', JSON.parse(results));
console.log(results);
        //returnMsg.data = JSON.parse(results);
        returnMsg.data = results;
        res.json(returnMsg);
        res.end();
    });
});

app.listen(port, function(){
    console.log('Webserver is listening at port :', port);
});
