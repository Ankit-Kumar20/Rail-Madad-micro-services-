const mongoose = require('mongoose');

const connectDB = async () => {
    try {
      await mongoose.connect('mongodb+srv://admin:deecc%4016@cluster0.cou91a0.mongodb.net/categorizationDB');
      console.log('MongoDB connected');
    } catch (error) {
      console.error(error.message);
      process.exit(1);
    }
  }

module.exports = connectDB;