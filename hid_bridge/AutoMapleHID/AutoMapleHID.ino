#include <Keyboard.h>

static const unsigned long SERIAL_BAUD = 115200;
static const char* PROTOCOL = "AUTO_MAPLE_HID_V1";

String line;

uint8_t keyCode(const String& name) {
  if (name == "LEFT") return KEY_LEFT_ARROW;
  if (name == "RIGHT") return KEY_RIGHT_ARROW;
  if (name == "UP") return KEY_UP_ARROW;
  if (name == "DOWN") return KEY_DOWN_ARROW;
  if (name == "SHIFT") return KEY_LEFT_SHIFT;
  if (name == "SPACE") return ' ';
  if (name == "Z") return 'z';
  if (name == "A") return 'a';
  return 0;
}

void emitReady() {
  Serial.print("READY ");
  Serial.println(PROTOCOL);
}

void processLine(String cmd) {
  cmd.trim();
  if (!cmd.length()) return;

  if (cmd == "PING") {
    Serial.print("PONG ");
    Serial.println(PROTOCOL);
    return;
  }

  if (cmd == "RELEASE_ALL") {
    Keyboard.releaseAll();
    Serial.println("OK");
    return;
  }

  int firstSpace = cmd.indexOf(' ');
  if (firstSpace < 0) {
    Serial.println("ERR command");
    return;
  }

  String op = cmd.substring(0, firstSpace);
  String rest = cmd.substring(firstSpace + 1);
  rest.trim();

  if (op == "KD" || op == "KU") {
    uint8_t code = keyCode(rest);
    if (!code) {
      Serial.println("ERR key");
      return;
    }
    if (op == "KD") Keyboard.press(code);
    else Keyboard.release(code);
    Serial.println("OK");
    return;
  }

  if (op == "PRESS") {
    int secondSpace = rest.indexOf(' ');
    if (secondSpace < 0) {
      Serial.println("ERR press");
      return;
    }
    String keyName = rest.substring(0, secondSpace);
    String msText = rest.substring(secondSpace + 1);
    keyName.trim();
    msText.trim();
    uint8_t code = keyCode(keyName);
    long holdMs = msText.toInt();
    if (!code || holdMs < 1 || holdMs > 2000) {
      Serial.println("ERR press args");
      return;
    }
    Keyboard.press(code);
    delay((unsigned long)holdMs);
    Keyboard.release(code);
    Serial.println("OK");
    return;
  }

  Serial.println("ERR command");
}

void setup() {
  Serial.begin(SERIAL_BAUD);
  Keyboard.begin();
  delay(800);
  emitReady();
}

void loop() {
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\r') continue;
    if (c == '\n') {
      processLine(line);
      line = "";
    } else if (line.length() < 96) {
      line += c;
    } else {
      line = "";
      Serial.println("ERR line too long");
    }
  }
}
