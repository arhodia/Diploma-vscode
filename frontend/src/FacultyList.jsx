import React, { useState } from "react";
import { RadioGroup, FormControlLabel, Radio } from '@mui/material';
import { 
  Select, MenuItem, InputLabel, FormControl, Button, Box, Typography, Alert, CircularProgress, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Paper
} from "@mui/material";

const STATIC_OPTIONS = [
 'business products   services',
 'consumer products   services',
 'it management',
 'real estate',
 'financial services',
 'engineering',
 'security',
 'logistics   transportation', 
 'insurance',
 'telecommunications',
 'manufacturing', 
 'travel   hospitality', 
 'software', 
 'construction',
 'environmental services', 
 'health', 
 'education', 
 'advertising   marketing', 
 'human resources', 
 'food   beverage', 
 'government services',
 'media', 
 'energy',
 'retail',
 'it system development', 
 'it services', 
 'computer hardware'
];

export default function FacultyList() {
  const [selectedOption, setSelectedOption] = useState('');
  const [uploadFile, setUploadFile] = useState(null);
  const [fileType, setFileType] = useState(null);
  const [loading, setLoading] = useState(false);
  const [uploadMessage, setUploadMessage] = useState({ type: '', text: '' });
  const [data, setData] = useState([]);

  // Dropdown option change
  const handleOptionChange = (e) => {
    setSelectedOption(e.target.value);
  };

  // File select
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    setUploadFile(file);
  };


  // Radio button choice for file type
  const handleFileTypeChange = (e) => {   
    setFileType(e.target.value);
  };

  // Upload (POST)
  const handleUpload = async () => {
    if (!selectedOption || !uploadFile) {
      setUploadMessage({type: 'error', text: 'Επίλεξε αρχείο ΚΑΙ πεδίο εργασίας!'});
      return;
    }

    setLoading(true);
    setUploadMessage({ type: '', text: '' });

    const formData = new FormData();
    formData.append('file', uploadFile); // μόνο ένα αρχείο!
    formData.append('selected_option', selectedOption);
    formData.append('file_type', fileType);

    try {
      const response = await fetch('http://127.0.0.1:5000/api/upload_files', {
        method: 'POST',
        body: formData
      });

      const result = await response.json();

      if (response.ok) {
        setUploadMessage({ type: 'success', text: result.message || 'Αρχείο ανέβηκε επιτυχώς!' });
        setData(result); // μόνο μετά το POST
        setUploadFile(null);
        document.getElementById('file-upload').value = '';
      } else {
        setUploadMessage({ type: 'error', text: result.error || 'Σφάλμα κατά το ανέβασμα.' });
      }
    } catch (error) {
      setUploadMessage({ type: 'error', text: 'Σφάλμα σύνδεσης με τον server' });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ width: "80%", margin: "0 auto", paddingTop: "2em" }}>
      <h2 style={{ textAlign: "center" }}>Faculty Dataset</h2>
      {/* Dropdown */}
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
      <Box sx={{ marginTop: 3, marginBottom: 3, padding: 3, border: '1px solid #ddd', borderRadius: 2, backgroundColor: '#f9f9f9' }}>
        <Typography variant="h6" gutterBottom>Ανέβασμα Αρχείου</Typography>
        <Box sx={{ marginBottom: 2 }}>
          <input
            accept=".csv"
            type="file"
            style={{ display: "none" }}
            id="file-upload"
            onChange={handleFileChange}
          />
          <label htmlFor="file-upload">
            <Button variant="outlined" component="span" color="primary" sx={{ marginRight: 2 }}>
              Upload File
            </Button>
          </label>
          {uploadFile && (
            <Typography variant="body2" component="span" color="success.main">
              ✓ {uploadFile.name}
            </Typography>
          )}
        </Box>
        {/* Submit Button */}
        <Button 
          variant="contained" 
          color="success" 
          onClick={handleUpload}
          disabled={loading || !uploadFile || !selectedOption}
          sx={{ marginTop: 2 }}
        >
          {loading ? <CircularProgress size={24} /> : 'Αποστολή Αρχείου'}
        </Button>
        <FormControl component="fieldset">
        <RadioGroup
          row
          value={fileType || ''}  // για controlled RadioGroup
          onChange={handleFileTypeChange}
          name="file_type"
        >
          <FormControlLabel value="start-up" control={<Radio />} label="Start-up" />
          <FormControlLabel value="researcher" control={<Radio />} label="Researcher" />
        </RadioGroup>
        </FormControl>
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
      {data.length > 0 && (
      <TableContainer component={Paper} style={{ marginTop: "2em" }}>
        <Table>
          <TableHead>
          <TableRow>
            <TableCell><strong>Όνομα - Επώνυμο / Εταιρεία</strong></TableCell>
            <TableCell><strong>Προφίλ</strong></TableCell>
            <TableCell><strong>Ερευνητικό Πεδίο</strong></TableCell>
            <TableCell><strong>Πεδίο</strong></TableCell>
          </TableRow>
          </TableHead>
          <TableBody>
            {data.map((row, idx) => (
              <TableRow key={idx}>
                <TableCell>{row.name} - {row.surname} / {row.company_name}</TableCell>
                <TableCell>{row.profile}</TableCell>
                <TableCell>{row.topic_merged_norm}</TableCell>
                <TableCell>{row.cluster}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>)}
    </div>
  );
}