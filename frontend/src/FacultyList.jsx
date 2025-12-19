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
  const [fileType, setFileType] = useState(null);
  const [loading, setLoading] = useState(false);
  const [uploadMessage, setUploadMessage] = useState({ type: '', text: '' });
  const [results, setResults] = useState([]);
  const [recommendations, setRecommendations] = useState([]);

  // Dropdown option change
  const handleOptionChange = (e) => {
    setSelectedOption(e.target.value);
  };


  // Radio button choice for file type
  const handleFileTypeChange = (e) => {   
    setFileType(e.target.value);
  };

  // Upload (POST)
  const handleUpload = async () => {
    if (!selectedOption ) {
      setUploadMessage({type: 'error', text: 'Επίλεξε αρχείο ΚΑΙ πεδίο εργασίας!'});
      return;
    }

    setLoading(true);
    setUploadMessage({ type: '', text: '' });

    const formData = new FormData();
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
        setResults(result.results || []);
        setRecommendations(result.recommendations || []);// μόνο μετά το POST
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
      {/* Submit Button */}
        <Button 
          variant="contained" 
          color="success" 
          onClick={handleUpload}
          disabled={loading || !selectedOption}
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
      {results.length > 0 && (
  <TableContainer component={Paper} style={{ marginTop: "2em" }}>
    <Typography variant="h6" sx={{ padding: 2, fontWeight: "bold" }}>
      Κορυφαία Αποτελέσματα
    </Typography>

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
        {results.map((row, idx) => (
          <TableRow key={idx}>
            <TableCell>{row.name} - {row.surname} / {row.company_name}</TableCell>
            <TableCell>{row.profile}</TableCell>
            <TableCell>{row.topic_merged_norm}</TableCell>
            <TableCell>{row.cluster}</TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  </TableContainer>
)}


{/* Προτεινόμενα Αποτελέσματα */}
{recommendations.length > 0 && (
  <TableContainer component={Paper} style={{ marginTop: "2em" }}>
    <Typography variant="h6" sx={{ padding: 2, fontWeight: "bold" }}>
      Προτεινόμενα Αποτελέσματα
    </Typography>

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
        {recommendations.map((row, idx) => (
          <TableRow key={idx}>
            <TableCell>{row.name} - {row.surname} / {row.company_name}</TableCell>
            <TableCell>{row.profile}</TableCell>
            <TableCell>{row.topic_merged_norm}</TableCell>
            <TableCell>{row.cluster}</TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  </TableContainer>
)}
    </div>
  );
}