import React, { useEffect, useState } from "react";
import { 
  Table, TableBody, TableCell, TableContainer, TableHead, TableRow, 
  Paper, 
  Button, 
  Select, 
  MenuItem, 
  InputLabel, 
  FormControl,
  Box,
  Typography,
  Alert,
  CircularProgress
} from "@mui/material";

const STATIC_OPTIONS = [
  "it management ",
  "environmental services ",
  "real estate",
  "education",
  "security",
  "travel   hospitality",
  "financial services",
  "consumer products   services",
  "energy",
  "computer hardware",
  "software",
  "government services",
  "it services ",
  "manufacturing",
  "media",
  "advertising   marketing",
  "retail",
  "human resources",
  "health",
  "it system development"
];

export default function FacultyList() {
  const [data, setData] = useState([]);
  const [selectedOption, setSelectedOption] = useState('');
  const [researchFile, setResearchFile] = useState(null);
  const [startupFile, setStartupFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [uploadMessage, setUploadMessage] = useState({ type: '', text: '' });

  // Fetch initial data (GET)
  useEffect(() => {
    fetch('http://127.0.0.1:5000/')
      .then(response => response.json())
      .then(data => setData(data))
      .catch(error => console.error('Error fetching data:', error));
  }, []);

  // Handle Research File selection
  const handleResearchFileChange = (e) => {
    const file = e.target.files[0];
    setResearchFile(file);
    console.log('Research file selected:', file?.name);
  };

  // Handle Startup File selection
  const handleStartupFileChange = (e) => {
    const file = e.target.files[0];
    setStartupFile(file);
    console.log('Startup file selected:', file?.name);
  };

  // Handle dropdown option change
  const handleOptionChange = (e) => {
    setSelectedOption(e.target.value);
  };

  // Handle Upload (POST)
  const handleUpload = async () => {
    // Validation
    if (!researchFile || !startupFile) {
      setUploadMessage({
        type: 'error',
        text: 'Παρακαλώ επίλεξε και τα δύο αρχεία (Research και Startup)'
      });
      return;
    }

    setLoading(true);
    setUploadMessage({ type: '', text: '' });

    // Create FormData object
    const formData = new FormData();
    formData.append('research_file', researchFile);
    formData.append('startup_file', startupFile);

    try {
      const response = await fetch('http://127.0.0.1:5000/api/upload_files', {
        method: 'POST',
        body: formData,
      });

      const result = await response.json();

      if (response.ok) {
        setUploadMessage({
          type: 'success',
          text: result.message || 'Τα αρχεία ανέβηκαν επιτυχώς!'
        });
        
      
        setData(result);
        // Reset file inputs
        setResearchFile(null);
        setStartupFile(null);
        document.getElementById('research-file-upload').value = '';
        document.getElementById('startup-file-upload').value = '';
      } else {
        setUploadMessage({
          type: 'error',
          text: result.error || 'Σφάλμα κατά το ανέβασμα των αρχείων'
        });
      }
    } catch (error) {
      console.error('Upload error:', error);
      setUploadMessage({
        type: 'error',
        text: 'Σφάλμα σύνδεσης με τον server'
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ width: "80%", margin: "0 auto", paddingTop: "2em" }}>
      <h2 style={{ textAlign: "center" }}>Faculty Dataset</h2>
      
      {/* Dropdown Selection */}
      <FormControl fullWidth margin="normal">
        <InputLabel id="dropdown-label">Select Option</InputLabel>
        <Select
          labelId="dropdown-label"
          value={selectedOption}
          label="Select Option"
          onChange={handleOptionChange}
        >
          {STATIC_OPTIONS.map(option => (
            <MenuItem key={option} value={option}>{option}</MenuItem>
          ))}
        </Select>
      </FormControl>

      {/* File Upload Section */}
      <Box sx={{ 
        marginTop: 3, 
        marginBottom: 3, 
        padding: 3, 
        border: '1px solid #ddd', 
        borderRadius: 2,
        backgroundColor: '#f9f9f9'
      }}>
        <Typography variant="h6" gutterBottom>
          Ανέβασμα Αρχείων
        </Typography>

        {/* Research File Upload */}
        <Box sx={{ marginBottom: 2 }}>
          <input
            accept=".csv"
            type="file"
            style={{ display: "none" }}
            id="research-file-upload"
            onChange={handleResearchFileChange}
          />
          <label htmlFor="research-file-upload">
            <Button 
              variant="outlined" 
              component="span" 
              color="primary"
              sx={{ marginRight: 2 }}
            >
              Upload Research File
            </Button>
          </label>
          {researchFile && (
            <Typography variant="body2" component="span" color="success.main">
              ✓ {researchFile.name}
            </Typography>
          )}
        </Box>

        {/* Startup File Upload */}
        <Box sx={{ marginBottom: 2 }}>
          <input
            accept=".csv"
            type="file"
            style={{ display: "none" }}
            id="startup-file-upload"
            onChange={handleStartupFileChange}
          />
          <label htmlFor="startup-file-upload">
            <Button 
              variant="outlined" 
              component="span" 
              color="primary"
              sx={{ marginRight: 2 }}
            >
              Upload Startup File
            </Button>
          </label>
          {startupFile && (
            <Typography variant="body2" component="span" color="success.main">
              ✓ {startupFile.name}
            </Typography>
          )}
        </Box>

        {/* Submit Button */}
        <Button 
          variant="contained" 
          color="success" 
          onClick={handleUpload}
          disabled={loading || !researchFile || !startupFile}
          sx={{ marginTop: 2 }}
        >
          {loading ? <CircularProgress size={24} /> : 'Αποστολή Αρχείων'}
        </Button>

        {/* Upload Message */}
        {uploadMessage.text && (
          <Alert 
            severity={uploadMessage.type} 
            sx={{ marginTop: 2 }}
            onClose={() => setUploadMessage({ type: '', text: '' })}
          >
            {uploadMessage.text}
          </Alert>
        )}
      </Box>

      {/* Data Table */}
      <TableContainer component={Paper} style={{ marginTop: "2em" }}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell><strong>Cluster</strong></TableCell>
              <TableCell><strong>Count</strong></TableCell>
              <TableCell><strong>Topic</strong></TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {data.length > 0 ? (
              data.map((row, idx) => (
                <TableRow key={idx}>
                  <TableCell>{row.cluster}</TableCell>
                  <TableCell>{row.count}</TableCell>
                  <TableCell>{row.topic_merged_norm}</TableCell>
                </TableRow>
              ))
            ) : (
              <TableRow>
                <TableCell colSpan={3} align="center">
                  Δεν υπάρχουν δεδομένα
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </TableContainer>
    </div>
  );
}
