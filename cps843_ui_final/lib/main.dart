import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'dart:io';
import 'package:image_picker/image_picker.dart';
import 'package:flutter_spinkit/flutter_spinkit.dart';
import 'package:just_audio/just_audio.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:flutter/services.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: HomePage(),
    );
  }
}

class HomePage extends StatefulWidget {
  @override
  _HomePageState createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  File? _image;
  String _caption = "";
  bool _loading = false;
  bool _showPerformanceMetrics = false;
  bool _showOcrOutput = false;
  String _ocrText = "";
  String _performanceMetrics = "";
  late AudioPlayer _audioPlayer;
  late FlutterTts _flutterTts;

  // Controller for the text field
  TextEditingController _textController = TextEditingController();

  @override
  void initState() {
    super.initState();
    _audioPlayer = AudioPlayer();
    _flutterTts = FlutterTts();
  }

  Future<void> _pickImage() async {
    try {
      final picker = ImagePicker();
      final pickedFile = await picker.pickImage(source: ImageSource.gallery);

      if (pickedFile != null) {
        setState(() {
          _image = File(pickedFile.path);
          _caption = "";
          _ocrText = "";
          _showOcrOutput = false;
        });
      } else {
        print("");
      }
    } catch (e) {
      print("Error picking image: $e");
    }
  }

  Future<void> _uploadImage(File image) async {
    setState(() {
      _loading = true;
    });

    try {
      final request = http.MultipartRequest(
          'POST', Uri.parse('http://127.0.0.1:5004/upload'));
      request.files.add(await http.MultipartFile.fromPath('image', image.path));

      final response = await request.send();
      if (response.statusCode == 200) {
        final responseData = await http.Response.fromStream(response);
        final data = json.decode(responseData.body);
        setState(() {
          _caption = data['caption'];
          _loading = false;
        });
      } else {
        setState(() {
          _caption = 'Failed to generate caption';
          _loading = false;
        });
      }
    } catch (e) {
      setState(() {
        _caption = 'Failed to upload image: $e';
        _loading = false;
      });
    }
  }

  Future<void> _performOCR(File image) async {
    setState(() {
      _loading = true;
    });

    try {
      final request =
          http.MultipartRequest('POST', Uri.parse('http://127.0.0.1:5004/ocr'));
      request.files.add(await http.MultipartFile.fromPath('image', image.path));

      final response = await request.send();
      if (response.statusCode == 200) {
        final responseData = await http.Response.fromStream(response);
        final data = json.decode(responseData.body);
        setState(() {
          _ocrText = data['ocr_text'] ?? 'No text found in the image';
          _loading = false;
        });

        // Show the OCR result in a dialog
        _showOcrDialog(_ocrText);
      } else {
        setState(() {
          _ocrText = 'Failed to perform OCR';
          _loading = false;
        });
        _showOcrDialog(_ocrText); // Show failure message in dialog
      }
    } catch (e) {
      setState(() {
        _ocrText = 'Error performing OCR: $e';
        _loading = false;
      });
      _showOcrDialog(_ocrText); // Show error message in dialog
    }
  }

  void _showOcrDialog(String ocrText) {
    TextEditingController _controller = TextEditingController(text: ocrText);

    showDialog(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          title: Text('OCR Output'),
          content: SingleChildScrollView(
            child: TextField(
              controller: _controller, // Use the controller for text input
              maxLines: null, // Allows the text to wrap and be multiline
              style: TextStyle(fontSize: 16),
              decoration: InputDecoration(
                border: OutlineInputBorder(),
                hintText: 'Edit OCR Text',
              ),
            ),
          ),
          actions: <Widget>[
            TextButton(
              onPressed: () {
                _flutterTts
                    .stop(); // Stop any ongoing speech before closing the dialog
                Navigator.of(context).pop(); // Close the dialog
              },
              child: Text('Close'),
            ),
            TextButton(
              onPressed: () {
                _speakOcrText(_controller.text); // Speak the editable OCR text
              },
              child: Text('Listen to OCR Text'),
            ),
            TextButton(
              onPressed: () {
                Clipboard.setData(ClipboardData(text: _controller.text));
                ScaffoldMessenger.of(context).showSnackBar(
                  SnackBar(content: Text('OCR Text copied to clipboard!')),
                );
              },
              child: Text('Copy'),
            ),
          ],
        );
      },
    );
  }

// Method to show the Performance Metrics in a dialog
  void _showPerformanceMetricsDialog(String performanceMetrics) {
    showDialog(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          title: Text('Performance Metrics'),
          content: SingleChildScrollView(
            child: Text(
              performanceMetrics,
              style: TextStyle(fontSize: 16),
            ),
          ),
          actions: <Widget>[
            TextButton(
              onPressed: () {
                Navigator.of(context).pop(); // Close the dialog
              },
              child: Text('Close'),
            ),
          ],
        );
      },
    );
  }

// Update _fetchPerformanceMetrics method to call _showPerformanceMetricsDialog
  Future<void> _fetchPerformanceMetrics() async {
    setState(() {
      _loading = true;
      _showPerformanceMetrics = true;
    });

    try {
      final response =
          await http.get(Uri.parse('http://127.0.0.1:5004/performance'));

      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        setState(() {
          _performanceMetrics = '''
        Time Taken: ${data['time_taken']} seconds
        Memory Used: ${data['memory_used']} MB
        BLEU Score: ${data['bleu_score']}
        ROUGE Score: ${data['rouge_score']['rouge1']}
        METEOR Score: ${data['meteor_score']}
        ''';
          _loading = false;
        });

        // Show the performance metrics in a dialog
        _showPerformanceMetricsDialog(_performanceMetrics);
      } else {
        setState(() {
          _performanceMetrics = 'Failed to fetch performance metrics';
          _loading = false;
        });
        _showPerformanceMetricsDialog(
            _performanceMetrics); // Show failure message in dialog
      }
    } catch (e) {
      setState(() {
        _performanceMetrics = 'Error fetching performance metrics: $e';
        _loading = false;
      });
      _showPerformanceMetricsDialog(
          _performanceMetrics); // Show error message in dialog
    }
  }

  Future<void> _speakCaption(String text) async {
    await _flutterTts.setLanguage("en-US");
    await _flutterTts.setPitch(1.0);
    await _flutterTts.setSpeechRate(0.5);

    var result = await _flutterTts.speak(text);
    if (result == 1) {
      print("Speaking: $text");
    }
  }

  Future<void> _speakOcrText(String ocrText) async {
    if (ocrText.isNotEmpty) {
      await _flutterTts.setLanguage("en-US");
      await _flutterTts.setPitch(1.0);
      await _flutterTts.setSpeechRate(0.5);

      var result = await _flutterTts.speak(ocrText);
      if (result == 1) {
        print("Speaking OCR text: $ocrText");
      }
    }
  }

  @override
  void dispose() {
    _audioPlayer.dispose();
    _flutterTts.stop();
    _textController.dispose(); // Dispose the controller when done
    super.dispose();
  }

  void _closePerformanceMetrics() {
    setState(() {
      _showPerformanceMetrics = false;
      _performanceMetrics = "";
    });
  }

  void _closeOcrOutput() {
    setState(() {
      _showOcrOutput = false;
      _ocrText = "";
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Image Caption Generator & OCR'),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: SingleChildScrollView(
          // Wrap the body with SingleChildScrollView
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: <Widget>[
              _image == null
                  ? Text('No image selected.', style: TextStyle(fontSize: 18))
                  : Column(
                      children: [
                        Image.file(_image!, height: 300),
                        SizedBox(height: 10), // Space between image and caption
                        _caption.isNotEmpty
                            ? Text(
                                _caption,
                                textAlign: TextAlign.center,
                                style: TextStyle(fontSize: 30),
                              )
                            : SizedBox.shrink(),
                      ],
                    ),
              SizedBox(height: 20),
              ElevatedButton(
                onPressed: _pickImage,
                child: Text('Pick Image'),
              ),
              SizedBox(height: 20),

              // Text Field for user input
              TextField(
                controller: _textController,
                decoration: InputDecoration(
                  labelText: 'Input Accurate Caption',
                  border: OutlineInputBorder(),
                ),
              ),
              SizedBox(
                  height: 20), // Space between text field and other widgets

              // Button to submit caption
              ElevatedButton(
                onPressed: _submitCaption,
                child: Text('Submit Caption'),
              ),
              SizedBox(height: 20), // Space after button

              _loading
                  ? SpinKitCircle(color: Colors.blue, size: 50.0)
                  : Center(
                      child: Column(
                        children: [
                          ElevatedButton(
                            onPressed: _image == null
                                ? null
                                : () => _uploadImage(_image!),
                            child: Text('Generate Caption'),
                          ),
                          SizedBox(height: 10),
                          ElevatedButton(
                            onPressed: _image == null
                                ? null
                                : () => _performOCR(_image!),
                            child: Text('Perform OCR'),
                          ),
                          SizedBox(height: 10),
                          ElevatedButton(
                            onPressed: _fetchPerformanceMetrics,
                            child: Text('View Performance Metrics'),
                          ),
                        ],
                      ),
                    ),
              SizedBox(height: 20),
              ElevatedButton(
                onPressed:
                    _caption.isNotEmpty ? () => _speakCaption(_caption) : null,
                child: Text('Listen to Audio'),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Future<void> _submitCaption() async {
    final String userInput = _textController.text;

    if (userInput.isNotEmpty) {
      // Log the user input for debugging
      print("User input: $userInput");

      // Send the user input to the Flask server
      try {
        final response = await http.post(
          Uri.parse('http://127.0.0.1:5004/submit_caption'),
          headers: {'Content-Type': 'application/json'},
          body: json.encode({'user_caption': userInput}),
        );

        if (response.statusCode == 200) {
          final data = json.decode(response.body);
          // Handle the response, for example, show the captions or performance metrics
          setState(() {
            _caption = data['generated_caption'];
            _performanceMetrics = '''
            Time Taken: ${data['performance']['time_taken']} seconds
            Memory Used: ${data['performance']['memory_used']} MB
            BLEU Score: ${data['performance']['bleu_score']}
            ROUGE Score: ${data['performance']['rouge_score']['rouge1']}
            METEOR Score: ${data['performance']['meteor_score']}
          ''';
          });
        } else {
          // If the server response is not 200, show an error message
          setState(() {
            _caption = 'Failed to submit caption';
          });
        }
      } catch (e) {
        setState(() {
          _caption = 'Error submitting caption: $e';
        });
      }
    } else {
      print("User input is empty.");
    }
  }
}
