const express = require('express')
const mongoose = require('mongoose')
const model = require('./DB/Models/categorization_model')

const router = express.Router();

router.use(express.json())

router.get('/', async function(req, res){
    const complaint_id = req.body.complaint_id;
    const category = req.body.category;
    const train_no = req.body.train_no;
    const Boggy_no = req.body.Boggy_no;


})

//CRUD opertions
router.post('/', async function(req, res){
    const complaint_id = req.body.complaint_id;
    const category = req.body.category;
    const train_no = req.body.train_no;
    const Boggy_no = req.body.Boggy_no;

    model.update({
        complaint_id: complaint_id,
        category: category,
        train_no: train_no,
        Boggy_no: Boggy_no
    })

})