import processing.video.*;
import java.io.IOException;

PImage img;
Capture cam;
String[] lines;
Table table;
float x = 0;
float y = 500;
boolean overButton = false;

void setup() {
  background(50);
  size(1366, 768);
  img = loadImage("data/output/mood.jpg");
  table = loadTable("data/output/music.cvs", "csv");
  String[] cameras = Capture.list();
  
  fill(255);          // Set the fill color to black
  textSize(24);     // Set the text size
  
  if (cameras.length == 0) {
    println("There are no cameras available for capture.");
    exit();
  } else {
    println("Available cameras:");
    for (int i = 0; i < cameras.length; i++) {
      println(cameras[i]);
    }
    // The camera can be initialized directly using an 
    // element from the array returned by list():
    cam = new Capture(this, cameras[0]);
    cam.start();     
  }      
}

void draw() {
  if (cam.available() == true) {
    cam.read();
  }
  set(0,0, cam);
    if (overButton == true) {
    fill(255);
  } else {
    noFill();
  }
  rect(105+ width/2+400, 60, 75, 75);
  line(135+ width/2+400, 105, 155+ width/2+400, 85);
  line(140+ width/2+400, 85, 155+ width/2+400, 85);
  line(155+ width/2+400, 85, 155+ width/2+400, 100);
}

void mousePressed() {
  if (overButton) { 
    link("https://open.spotify.com/track/" + table.getRow(6).getString(3));
  }
  else{
    background(50);
    y = 500;
    cam.save("data/input/mood.jpg");
      // Specify the command to run your Python script
    String cmd = "python " + dataPath("moodMusic.py");
    try {
      Process process = Runtime.getRuntime().exec(cmd);
      process.waitFor();
      println("Python script executed.");
    } catch (Exception e) {
      e.printStackTrace();
      println("Error running Python script.");
    }
    lines = loadStrings("data/output/mood.txt");
    img = loadImage("data/output/mood.jpg");
    img.resize(300,300);
    image(img, width/2, 0);
    String mood = lines[0];
    text("Mood: "+mood, width / 2 + 100, height / 2 - 25);
    table = loadTable("data/output/music.cvs", "csv");
    text("Recommending: " + table.getRow(1).getString(18) + " music", width / 2, height / 2);
  //for (TableRow row : table.rows()) {
    for (int i = 0; i < 12; i++) {
      TableRow row = table.getRow(i);
      String name = row.getString(0);
      String album = row.getString(1);
      String artist = row.getString(2);
      
      text(name + "--" + album + "--" + artist, x, y);
      y += 25;
    }
  }
}

void mouseMoved() { 
  checkButtons(); 
}
  
void mouseDragged() {
  checkButtons(); 
}

void checkButtons() {
  if (mouseX > 105+ width/2+400 && mouseX < 180+ width/2+400 && mouseY > 60 && mouseY <135) {
    overButton = true;   
  } else {
    overButton = false;
  }
}
