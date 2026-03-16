"""Unit tests for model2obs.cli.download_crocolake module.

Tests cover database code name generation, file download operations,
ZIP extraction, and error handling for network operations.
"""

from pathlib import Path
from unittest.mock import Mock, patch, mock_open
import io
import zipfile

import pytest
import responses
from requests.exceptions import HTTPError, ConnectionError, Timeout

from model2obs.cli import download_crocolake


class TestGetDbCodename:
    """Test suite for get_db_codename() function."""
    
    def test_crocolake_phy_qc(self):
        """Test codename for CrocoLake PHY with QC."""
        result = download_crocolake.get_db_codename("CROCOLAKE", "PHY", qc=True)
        assert result == "0007_PHY_CROCOLAKE-QC-MERGED"
    
    def test_crocolake_bgc_qc(self):
        """Test codename for CrocoLake BGC with QC."""
        result = download_crocolake.get_db_codename("CROCOLAKE", "BGC", qc=True)
        assert result == "0007_BGC_CROCOLAKE-QC-MERGED"
    
    def test_crocolake_no_qc_raises_error(self):
        """Test CrocoLake without QC raises ValueError."""
        with pytest.raises(ValueError, match="CrocoLake database available only with QC"):
            download_crocolake.get_db_codename("CROCOLAKE", "PHY", qc=False)
    
    def test_argo_phy_qc(self):
        """Test codename for Argo PHY with QC."""
        result = download_crocolake.get_db_codename("ARGO", "PHY", qc=True)
        assert result == "1003_PHY_ARGO-QC"
    
    def test_argo_phy_no_qc(self):
        """Test codename for Argo PHY without QC (cloud)."""
        result = download_crocolake.get_db_codename("ARGO", "PHY", qc=False)
        assert result == "1011_PHY_ARGO-CLOUD"
    
    def test_argo_bgc_qc(self):
        """Test codename for Argo BGC with QC."""
        result = download_crocolake.get_db_codename("ARGO", "BGC", qc=True)
        assert result == "1003_BGC_ARGO-QC"
    
    def test_argo_bgc_no_qc(self):
        """Test codename for Argo BGC without QC (cloud)."""
        result = download_crocolake.get_db_codename("ARGO", "BGC", qc=False)
        assert result == "1011_BGC_ARGO-CLOUD"
    
    def test_invalid_db_type_raises_error(self):
        """Test invalid database type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid database type"):
            download_crocolake.get_db_codename("CROCOLAKE", "INVALID", qc=True)
    
    def test_dev_version_suffix(self):
        """Test dev=True adds -DEV suffix."""
        result = download_crocolake.get_db_codename("ARGO", "PHY", qc=True, dev=True)
        assert result == "1003_PHY_ARGO-QC-DEV"
    
    def test_dev_version_false(self):
        """Test dev=False returns standard codename."""
        result = download_crocolake.get_db_codename("ARGO", "PHY", qc=True, dev=False)
        assert result == "1003_PHY_ARGO-QC"


class TestDownloadFile:
    """Test suite for download_file() function."""
    
    @responses.activate
    def test_download_file_success(self, tmp_path):
        """Test successful file download with progress tracking."""
        url = "https://example.com/testfile.zip"
        local_file = tmp_path / "downloaded.zip"
        content = b"test file content"
        
        responses.add(
            responses.GET,
            url,
            body=content,
            status=200,
            headers={'content-length': str(len(content))}
        )
        
        with patch('model2obs.cli.download_crocolake.tqdm') as mock_tqdm:
            mock_tqdm.return_value.__enter__ = Mock(return_value=Mock())
            mock_tqdm.return_value.__exit__ = Mock(return_value=False)
            
            download_crocolake.download_file(url, str(local_file))
        
        assert local_file.exists()
        assert local_file.read_bytes() == content
    
    @responses.activate
    def test_download_file_404_error(self, tmp_path, capsys):
        """Test download handles 404 Not Found error."""
        url = "https://example.com/missing.zip"
        local_file = tmp_path / "downloaded.zip"
        
        responses.add(
            responses.GET,
            url,
            status=404
        )
        
        download_crocolake.download_file(url, str(local_file))
        
        captured = capsys.readouterr()
        assert "An error occurred" in captured.out
        assert not local_file.exists()
    
    @responses.activate
    def test_download_file_connection_error(self, tmp_path, capsys):
        """Test download handles connection errors."""
        url = "https://example.com/testfile.zip"
        local_file = tmp_path / "downloaded.zip"
        
        responses.add(
            responses.GET,
            url,
            body=ConnectionError("Connection refused")
        )
        
        download_crocolake.download_file(url, str(local_file))
        
        captured = capsys.readouterr()
        assert "An error occurred" in captured.out
    
    @responses.activate
    def test_download_file_no_content_length(self, tmp_path):
        """Test download works without content-length header."""
        url = "https://example.com/testfile.zip"
        local_file = tmp_path / "downloaded.zip"
        content = b"test content"
        
        responses.add(
            responses.GET,
            url,
            body=content,
            status=200
        )
        
        with patch('model2obs.cli.download_crocolake.tqdm') as mock_tqdm:
            mock_tqdm.return_value.__enter__ = Mock(return_value=Mock())
            mock_tqdm.return_value.__exit__ = Mock(return_value=False)
            
            download_crocolake.download_file(url, str(local_file))
        
        assert local_file.exists()
    
    @responses.activate
    def test_download_file_progress_updates(self, tmp_path):
        """Test download progress bar updates correctly."""
        url = "https://example.com/testfile.zip"
        local_file = tmp_path / "downloaded.zip"
        content = b"x" * 100000
        
        responses.add(
            responses.GET,
            url,
            body=content,
            status=200,
            headers={'content-length': str(len(content))}
        )
        
        with patch('model2obs.cli.download_crocolake.tqdm') as mock_tqdm:
            mock_bar = Mock()
            mock_tqdm.return_value.__enter__ = Mock(return_value=mock_bar)
            mock_tqdm.return_value.__exit__ = Mock(return_value=False)
            
            download_crocolake.download_file(url, str(local_file))
            
            assert mock_bar.update.called


class TestUnzipFile:
    """Test suite for unzip_file() function."""
    
    def test_unzip_file_basic(self, tmp_path):
        """Test basic ZIP file extraction."""
        zip_path = tmp_path / "test.zip"
        extract_to = tmp_path / "extracted"
        extract_to.mkdir()
        
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("file1.txt", "content1")
            zf.writestr("dir/file2.txt", "content2")
        
        download_crocolake.unzip_file(str(zip_path), str(extract_to))
        
        assert (extract_to / "file1.txt").exists()
        assert (extract_to / "dir" / "file2.txt").exists()
        assert (extract_to / "file1.txt").read_text() == "content1"
    
    def test_unzip_file_empty_archive(self, tmp_path):
        """Test extraction of empty ZIP archive."""
        zip_path = tmp_path / "empty.zip"
        extract_to = tmp_path / "extracted"
        extract_to.mkdir()
        
        with zipfile.ZipFile(zip_path, 'w'):
            pass
        
        download_crocolake.unzip_file(str(zip_path), str(extract_to))
        
        assert extract_to.exists()
    
    def test_unzip_file_missing_zip_raises_error(self, tmp_path):
        """Test unzip raises error for missing ZIP file."""
        zip_path = tmp_path / "missing.zip"
        extract_to = tmp_path / "extracted"
        
        with pytest.raises(FileNotFoundError):
            download_crocolake.unzip_file(str(zip_path), str(extract_to))
    
    def test_unzip_file_corrupted_zip_raises_error(self, tmp_path):
        """Test unzip raises error for corrupted ZIP file."""
        zip_path = tmp_path / "corrupted.zip"
        extract_to = tmp_path / "extracted"
        extract_to.mkdir()
        
        zip_path.write_bytes(b"not a zip file")
        
        with pytest.raises(zipfile.BadZipFile):
            download_crocolake.unzip_file(str(zip_path), str(extract_to))
    
    def test_unzip_file_preserves_structure(self, tmp_path):
        """Test unzip preserves directory structure."""
        zip_path = tmp_path / "structured.zip"
        extract_to = tmp_path / "extracted"
        extract_to.mkdir()
        
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("level1/level2/file.txt", "nested")
        
        download_crocolake.unzip_file(str(zip_path), str(extract_to))
        
        nested_file = extract_to / "level1" / "level2" / "file.txt"
        assert nested_file.exists()
        assert nested_file.read_text() == "nested"


class TestMain:
    """Test suite for main() CLI function."""
    
    @responses.activate
    @patch('model2obs.cli.download_crocolake.unzip_file')
    @patch('model2obs.cli.download_crocolake.tqdm')
    def test_main_crocolake_phy_qc(self, mock_tqdm, mock_unzip, tmp_path, monkeypatch):
        """Test main function with CrocoLake PHY QC database."""
        mock_tqdm.return_value.__enter__ = Mock(return_value=Mock())
        mock_tqdm.return_value.__exit__ = Mock(return_value=False)
        
        destination = tmp_path / "downloads"
        
        test_args = [
            'prog',
            '-d', 'CROCOLAKE',
            '-t', 'PHY',
            '--qc',
            '--destination', str(destination)
        ]
        monkeypatch.setattr('sys.argv', test_args)
        
        url = download_crocolake.urls["0007_PHY_CROCOLAKE-QC-MERGED"]
        responses.add(
            responses.GET,
            url,
            body=b"fake zip content",
            status=200,
            headers={'content-length': '100'}
        )
        
        download_crocolake.main()
        
        expected_dir = destination / "0007_PHY_CROCOLAKE-QC-MERGED"
        assert expected_dir.exists()
        assert mock_unzip.called
    
    @patch('model2obs.cli.download_crocolake.download_file')
    @patch('model2obs.cli.download_crocolake.unzip_file')
    def test_main_conflicting_qc_flags_raises_error(self, mock_unzip, mock_download, monkeypatch):
        """Test main raises error when both --qc and --noqc are specified."""
        test_args = [
            'prog',
            '-d', 'ARGO',
            '-t', 'PHY',
            '--qc',
            '--noqc'
        ]
        monkeypatch.setattr('sys.argv', test_args)
        
        with pytest.raises(ValueError, match="Cannot specify both"):
            download_crocolake.main()
    
    @responses.activate
    @patch('model2obs.cli.download_crocolake.unzip_file')
    @patch('model2obs.cli.download_crocolake.tqdm')
    def test_main_default_qc_true(self, mock_tqdm, mock_unzip, tmp_path, monkeypatch):
        """Test main defaults to QC=True when no flag specified."""
        mock_tqdm.return_value.__enter__ = Mock(return_value=Mock())
        mock_tqdm.return_value.__exit__ = Mock(return_value=False)
        
        destination = tmp_path / "downloads"
        
        test_args = [
            'prog',
            '-d', 'ARGO',
            '-t', 'PHY',
            '--destination', str(destination)
        ]
        monkeypatch.setattr('sys.argv', test_args)
        
        url = download_crocolake.urls["1003_PHY_ARGO-QC"]
        responses.add(
            responses.GET,
            url,
            body=b"fake content",
            status=200,
            headers={'content-length': '100'}
        )
        
        download_crocolake.main()
        
        expected_dir = destination / "1003_PHY_ARGO-QC"
        assert expected_dir.exists()
    
    @responses.activate
    @patch('model2obs.cli.download_crocolake.unzip_file')
    @patch('model2obs.cli.download_crocolake.tqdm')
    def test_main_destination_trailing_slash(self, mock_tqdm, mock_unzip, tmp_path, monkeypatch):
        """Test main handles destination with trailing slash."""
        mock_tqdm.return_value.__enter__ = Mock(return_value=Mock())
        mock_tqdm.return_value.__exit__ = Mock(return_value=False)
        
        destination = str(tmp_path / "downloads") + "/"
        
        test_args = [
            'prog',
            '-d', 'ARGO',
            '-t', 'BGC',
            '--qc',
            '--destination', destination
        ]
        monkeypatch.setattr('sys.argv', test_args)
        
        url = download_crocolake.urls["1003_BGC_ARGO-QC"]
        responses.add(
            responses.GET,
            url,
            body=b"fake content",
            status=200,
            headers={'content-length': '100'}
        )
        
        download_crocolake.main()
        
        assert Path(destination).exists()
    
    @responses.activate
    @patch('model2obs.cli.download_crocolake.unzip_file')
    @patch('model2obs.cli.download_crocolake.tqdm')
    def test_main_cleanup_removes_zip(self, mock_tqdm, mock_unzip, tmp_path, monkeypatch, capsys):
        """Test main cleans up downloaded ZIP file after extraction."""
        mock_tqdm.return_value.__enter__ = Mock(return_value=Mock())
        mock_tqdm.return_value.__exit__ = Mock(return_value=False)
        
        destination = tmp_path / "downloads"
        
        test_args = [
            'prog',
            '-d', 'ARGO',
            '-t', 'PHY',
            '--noqc',
            '--destination', str(destination)
        ]
        monkeypatch.setattr('sys.argv', test_args)
        
        url = download_crocolake.urls["1011_PHY_ARGO-CLOUD"]
        responses.add(
            responses.GET,
            url,
            body=b"fake content",
            status=200,
            headers={'content-length': '100'}
        )
        
        download_crocolake.main()
        
        captured = capsys.readouterr()
        assert "Cleaning up..." in captured.out
        assert "Database setup complete" in captured.out
        
        zip_file = destination / "1011_PHY_ARGO-CLOUD" / "1011_PHY_ARGO-CLOUD.zip"
        assert not zip_file.exists()
