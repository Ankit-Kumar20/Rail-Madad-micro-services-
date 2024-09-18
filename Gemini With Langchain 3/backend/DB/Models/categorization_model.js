const mongoose = require('mongoose');
const connectDB = require('../database')

connectDB();

const catego_schema = new mongoose.Schema({
    complaint_id: Number,
    description: String,
    train_no: Number,
    Boggy_no: String,
    category: String
})

catego_model = mongoose.model("categorization_model", catego_schema);

module.exports = catego_model;