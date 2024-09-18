const express = require("express");
const model = require('./DB/database')
const jwt = require('jsonwebtoken')

const app = express();
const router = express.Router();

router.use(express.json())

router.post('/', async function(req, res){
    
})
