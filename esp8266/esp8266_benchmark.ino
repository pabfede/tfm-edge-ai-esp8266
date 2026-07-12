// ============================================================
// TFM — Benchmark de latencia de inferencia en ESP8266
// Técnica: Destilación + INT8
// Modelo: conv_ae_model_data.h (18352 bytes)
// Autor: Pablo Federico Martín Luna — UNIR MIA 2026
// ============================================================
// NOTA TÉCNICA: se usa Chirale_TensorFlowLite en lugar de
// esp-tflite-micro, ya que este último requiere ESP-IDF y solo
// soporta arquitecturas ESP32/S/C, incompatible con el core
// Arduino-ESP8266 (Xtensa LX106) usado en este proyecto.
//
// NOTA TÉCNICA 2: se usa MicroMutableOpResolver con los 11
// operadores reales del modelo (verificados por inspección del
// flatbuffer) en lugar de AllOpsResolver (128 operaciones, coste
// fijo de RAM ~30.7 KB medido empíricamente). Esto libera margen
// suficiente para tensor_arena.
//
// NOTA TÉCNICA 3: el modelo NO se declara con PROGMEM. Chirale_
// TensorFlowLite no implementa acceso alineado a flash en el
// parseo del flatbuffer, lo que provoca excepción
// LoadStoreErrorCause en Xtensa LX106 al usar PROGMEM.
// ============================================================
#include <Chirale_TensorFlowLite.h>
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "conv_ae_model_data.h"  // g_conv_ae_model, SIN PROGMEM

constexpr int kWindowSize = 24;
constexpr int kNSignals   = 6;

// Tamaño del tensor arena. Punto de partida empírico: ajustar
// en pasos de 2 KB si AllocateTensors() falla (aumentar) o si
// se busca optimizar el uso de RAM (reducir). Ver Anexo D para
// el método de medición de RAM real disponible por etapa.
constexpr int kArenaSize = 20480;  // 20 KB

alignas(16) uint8_t tensor_arena[kArenaSize];

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Resolver ajustado a los 11 operadores reales del modelo,
// verificados por inspección directa del flatbuffer con
// tf.lite.Interpreter (BUILTIN_WITHOUT_DEFAULT_DELEGATES).
static tflite::MicroMutableOpResolver<11> resolver;

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("\n=== TFM ESP8266 — Benchmark OE5 ===");
  Serial.println("Tecnica: Destilación + INT8");
  Serial.print("RAM libre inicial: ");
  Serial.print(ESP.getFreeHeap());
  Serial.println(" bytes");

  model = tflite::GetModel(g_conv_ae_model);

  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.print("ERROR: version incompatible. Modelo: ");
    Serial.print(model->version());
    Serial.print(" | TFLM: ");
    Serial.println(TFLITE_SCHEMA_VERSION);
    return;
  }

  // Registro explícito de los operadores reales del modelo.
  resolver.AddAdd();
  resolver.AddConcatenation();
  resolver.AddConv2D();
  resolver.AddExpandDims();
  resolver.AddLogistic();
  resolver.AddMaxPool2D();
  resolver.AddMul();
  resolver.AddReshape();
  resolver.AddShape();
  resolver.AddStridedSlice();
  resolver.AddTransposeConv();

  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kArenaSize
  );
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.print("ERROR: AllocateTensors fallo. RAM libre: ");
    Serial.print(ESP.getFreeHeap());
    Serial.println(" bytes. Aumentar kArenaSize.");
    return;
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.print("Modelo cargado. RAM libre tras init: ");
  Serial.print(ESP.getFreeHeap());
  Serial.println(" bytes");

  Serial.print("Shape entrada: (");
  Serial.print(input->dims->data[0]); Serial.print(", ");
  Serial.print(input->dims->data[1]); Serial.print(", ");
  Serial.print(input->dims->data[2]); Serial.println(")");
  Serial.print("Tipo entrada: ");
  Serial.println(input->type == kTfLiteInt8 ? "INT8 OK" : "ERROR tipo inesperado");

  // Ventana de prueba sintética (valor constante normalizado 0.5)
  for (int i = 0; i < kWindowSize * kNSignals; i++) {
    input->data.int8[i] = 64;
  }

  Serial.println("\nIniciando 100 inferencias de benchmark...");
  delay(200);

  unsigned long total_us = 0;
  const int N_RUNS = 100;
  for (int r = 0; r < N_RUNS; r++) {
    unsigned long t0 = micros();
    TfLiteStatus invoke_status = interpreter->Invoke();
    total_us += micros() - t0;
    if (invoke_status != kTfLiteOk) {
      Serial.print("ERROR: Invoke fallo en la iteracion ");
      Serial.println(r);
      return;
    }
  }

  float avg_ms = (float)(total_us / N_RUNS) / 1000.0f;
  Serial.println("\n=== RESULTADO OE5 ===");
  Serial.print("Inferencias ejecutadas: ");
  Serial.println(N_RUNS);
  Serial.print("Tiempo total: ");
  Serial.print(total_us / 1000.0f, 1);
  Serial.println(" ms");
  Serial.print("Latencia media por inferencia: ");
  Serial.print(avg_ms, 2);
  Serial.println(" ms <-- ANOTAR EN LA MEMORIA (Cap. 5.3)");
  Serial.print("Cumple restriccion <500 ms: ");
  Serial.println(avg_ms < 500.0f ? "SI" : "NO");
  Serial.print("RAM libre tras inferencia: ");
  Serial.print(ESP.getFreeHeap());
  Serial.println(" bytes");

  while (true) { delay(60000); }
}

void loop() {}
