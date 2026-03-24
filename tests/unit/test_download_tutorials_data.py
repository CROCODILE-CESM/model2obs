"""Unit tests for model2obs.cli.download_tutorials_data module.

Tests cover Zenodo API integration, file download operations,
ZIP extraction, and error handling for tutorial data downloads.
"""

from pathlib import Path
from unittest.mock import Mock, patch
import zipfile

import pytest
import responses
from requests.exceptions import HTTPError, ConnectionError

from model2obs.cli import download_tutorials_data


class TestFetchZenodoUrl:
    """Test suite for fetch_zenodo_url() function."""
    
    @responses.activate
    def test_fetch_zenodo_url_success(self):
        """Test successful fetching of Zenodo record URLs."""
        zenodo_id = 12345
        metadata_url = f"https://zenodo.org/api/records/{zenodo_id}"
        
        mock_response = {
            'files': [
                {
                    'key': 'file1.zip',
                    'links': {'self': 'https://zenodo.org/api/files/1/file1.zip'}
                },
                {
                    'key': 'file2.dat',
                    'links': {'self': 'https://zenodo.org/api/files/1/file2.dat'}
                }
            ]
        }
        
        responses.add(
            responses.GET,
            metadata_url,
            json=mock_response,
            status=200
        )
        
        urls, filenames = download_tutorials_data.fetch_zenodo_url(zenodo_id)
        
        assert len(urls) == 2
        assert len(filenames) == 2
        assert 'file1.zip' in filenames
        assert 'file2.dat' in filenames
        assert 'https://zenodo.org/api/files/1/file1.zip' in urls
    
    @responses.activate
    def test_fetch_zenodo_url_single_file(self):
        """Test fetching Zenodo record with single file."""
        zenodo_id = 12345
        metadata_url = f"https://zenodo.org/api/records/{zenodo_id}"
        
        mock_response = {
            'files': [
                {
                    'key': 'data.zip',
                    'links': {'self': 'https://zenodo.org/api/files/1/data.zip'}
                }
            ]
        }
        
        responses.add(
            responses.GET,
            metadata_url,
            json=mock_response,
            status=200
        )
        
        urls, filenames = download_tutorials_data.fetch_zenodo_url(zenodo_id)
        
        assert len(urls) == 1
        assert len(filenames) == 1
        assert filenames[0] == 'data.zip'
    
    @responses.activate
    def test_fetch_zenodo_url_404_error(self):
        """Test fetch raises error for non-existent Zenodo record."""
        zenodo_id = 99999
        metadata_url = f"https://zenodo.org/api/records/{zenodo_id}"
        
        responses.add(
            responses.GET,
            metadata_url,
            status=404
        )
        
        with pytest.raises(HTTPError):
            download_tutorials_data.fetch_zenodo_url(zenodo_id)
    
    @responses.activate
    def test_fetch_zenodo_url_empty_files(self):
        """Test fetching Zenodo record with no files."""
        zenodo_id = 12345
        metadata_url = f"https://zenodo.org/api/records/{zenodo_id}"
        
        mock_response = {'files': []}
        
        responses.add(
            responses.GET,
            metadata_url,
            json=mock_response,
            status=200
        )
        
        urls, filenames = download_tutorials_data.fetch_zenodo_url(zenodo_id)
        
        assert len(urls) == 0
        assert len(filenames) == 0
    
    @responses.activate
    def test_fetch_zenodo_url_connection_error(self):
        """Test fetch handles connection errors."""
        zenodo_id = 12345
        metadata_url = f"https://zenodo.org/api/records/{zenodo_id}"
        
        responses.add(
            responses.GET,
            metadata_url,
            body=ConnectionError("Network unreachable")
        )
        
        with pytest.raises(ConnectionError):
            download_tutorials_data.fetch_zenodo_url(zenodo_id)


class TestDownloadZenodoRecord:
    """Test suite for download_zenodo_record() function."""
    
    @responses.activate
    @patch('model2obs.cli.download_tutorials_data.download_file')
    def test_download_zenodo_record_success(self, mock_download, capsys):
        """Test successful download of Zenodo record files."""
        zenodo_id = 12345
        download_to = "/tmp/downloads/"
        
        metadata_url = f"https://zenodo.org/api/records/{zenodo_id}"
        mock_response = {
            'files': [
                {
                    'key': 'file1.zip',
                    'links': {'self': 'https://zenodo.org/api/files/1/file1.zip'}
                }
            ]
        }
        
        responses.add(
            responses.GET,
            metadata_url,
            json=mock_response,
            status=200
        )
        
        filenames = download_tutorials_data.download_zenodo_record(zenodo_id, download_to)
        
        assert filenames == ['file1.zip']
        mock_download.assert_called_once_with(
            'https://zenodo.org/api/files/1/file1.zip',
            '/tmp/downloads/file1.zip'
        )
        
        captured = capsys.readouterr()
        assert "Downloading" in captured.out
    
    @responses.activate
    @patch('model2obs.cli.download_tutorials_data.download_file')
    def test_download_zenodo_record_multiple_files(self, mock_download):
        """Test download of Zenodo record with multiple files."""
        zenodo_id = 12345
        download_to = "/tmp/downloads/"
        
        metadata_url = f"https://zenodo.org/api/records/{zenodo_id}"
        mock_response = {
            'files': [
                {
                    'key': 'file1.zip',
                    'links': {'self': 'https://zenodo.org/api/files/1/file1.zip'}
                },
                {
                    'key': 'file2.dat',
                    'links': {'self': 'https://zenodo.org/api/files/1/file2.dat'}
                }
            ]
        }
        
        responses.add(
            responses.GET,
            metadata_url,
            json=mock_response,
            status=200
        )
        
        filenames = download_tutorials_data.download_zenodo_record(zenodo_id, download_to)
        
        assert len(filenames) == 2
        assert mock_download.call_count == 2


class TestDownloadFile:
    """Test suite for download_file() function."""
    
    @responses.activate
    def test_download_file_success(self, tmp_path):
        """Test successful file download."""
        url = "https://example.com/testfile.dat"
        local_file = tmp_path / "downloaded.dat"
        content = b"test file content"
        
        responses.add(
            responses.GET,
            url,
            body=content,
            status=200,
            headers={'content-length': str(len(content))}
        )
        
        with patch('model2obs.cli.download_tutorials_data.tqdm') as mock_tqdm:
            mock_tqdm.return_value.__enter__ = Mock(return_value=Mock())
            mock_tqdm.return_value.__exit__ = Mock(return_value=False)
            
            download_tutorials_data.download_file(url, str(local_file))
        
        assert local_file.exists()
        assert local_file.read_bytes() == content
    
    @responses.activate
    def test_download_file_error_handling(self, tmp_path, capsys):
        """Test download handles HTTP errors gracefully."""
        url = "https://example.com/missing.dat"
        local_file = tmp_path / "downloaded.dat"
        
        responses.add(
            responses.GET,
            url,
            status=404
        )
        
        download_tutorials_data.download_file(url, str(local_file))
        
        captured = capsys.readouterr()
        assert "An error occurred" in captured.out
        assert not local_file.exists()


class TestUnzipFile:
    """Test suite for unzip_file() function."""
    
    def test_unzip_file_with_progress(self, tmp_path):
        """Test ZIP extraction with progress tracking."""
        zip_path = tmp_path / "test.zip"
        extract_to = tmp_path / "extracted"
        extract_to.mkdir()
        
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("file1.txt", "content1" * 1000)
            zf.writestr("dir/file2.txt", "content2" * 1000)
        
        with patch('model2obs.cli.download_tutorials_data.tqdm') as mock_tqdm:
            mock_tqdm.return_value.__enter__ = Mock(return_value=Mock())
            mock_tqdm.return_value.__exit__ = Mock(return_value=False)
            
            download_tutorials_data.unzip_file(str(zip_path), str(extract_to))
        
        assert (extract_to / "file1.txt").exists()
        assert (extract_to / "dir" / "file2.txt").exists()
    
    def test_unzip_file_creates_directories(self, tmp_path):
        """Test unzip creates nested directories as needed."""
        zip_path = tmp_path / "nested.zip"
        extract_to = tmp_path / "extracted"
        extract_to.mkdir()
        
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("a/b/c/file.txt", "nested")
        
        with patch('model2obs.cli.download_tutorials_data.tqdm') as mock_tqdm:
            mock_tqdm.return_value.__enter__ = Mock(return_value=Mock())
            mock_tqdm.return_value.__exit__ = Mock(return_value=False)
            
            download_tutorials_data.unzip_file(str(zip_path), str(extract_to))
        
        nested_file = extract_to / "a" / "b" / "c" / "file.txt"
        assert nested_file.exists()
        assert nested_file.read_text() == "nested"
    
    def test_unzip_file_handles_directories(self, tmp_path):
        """Test unzip correctly handles directory entries."""
        zip_path = tmp_path / "with_dirs.zip"
        extract_to = tmp_path / "extracted"
        extract_to.mkdir()
        
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("emptydir/", "")
            zf.writestr("emptydir/file.txt", "content")
        
        with patch('model2obs.cli.download_tutorials_data.tqdm') as mock_tqdm:
            mock_tqdm.return_value.__enter__ = Mock(return_value=Mock())
            mock_tqdm.return_value.__exit__ = Mock(return_value=False)
            
            download_tutorials_data.unzip_file(str(zip_path), str(extract_to))
        
        assert (extract_to / "emptydir").is_dir()
        assert (extract_to / "emptydir" / "file.txt").exists()


class TestMain:
    """Test suite for main() CLI function."""
    
    @responses.activate
    @patch('model2obs.cli.download_tutorials_data.unzip_file')
    @patch('model2obs.cli.download_tutorials_data.download_file')
    def test_main_default_destination(self, mock_download, mock_unzip, tmp_path, monkeypatch):
        """Test main with default destination folder."""
        test_args = ['prog']
        monkeypatch.setattr('sys.argv', test_args)
        
        with patch('os.makedirs'):
            with patch('os.remove'):
                metadata_url = "https://zenodo.org/api/records/19204393"
                mock_response = {
                    'files': [
                        {
                            'key': 'tutorial.zip',
                            'links': {'self': 'https://zenodo.org/api/files/1/tutorial.zip'}
                        }
                    ]
                }
                
                responses.add(
                    responses.GET,
                    metadata_url,
                    json=mock_response,
                    status=200
                )
                
                download_tutorials_data.main()
                
                assert mock_download.called
                assert mock_unzip.called
    
    @responses.activate
    @patch('model2obs.cli.download_tutorials_data.unzip_file')
    @patch('model2obs.cli.download_tutorials_data.download_file')
    def test_main_custom_destination(self, mock_download, mock_unzip, tmp_path, monkeypatch):
        """Test main with custom destination folder."""
        destination = tmp_path / "custom_dest"
        
        test_args = [
            'prog',
            '--destination', str(destination)
        ]
        monkeypatch.setattr('sys.argv', test_args)
        
        with patch('os.makedirs'):
            with patch('os.remove'):
                metadata_url = "https://zenodo.org/api/records/19204393"
                mock_response = {
                    'files': [
                        {
                            'key': 'data.zip',
                            'links': {'self': 'https://zenodo.org/api/files/1/data.zip'}
                        }
                    ]
                }
                
                responses.add(
                    responses.GET,
                    metadata_url,
                    json=mock_response,
                    status=200
                )
                
                download_tutorials_data.main()
                
                assert mock_download.called
    
    @responses.activate
    @patch('model2obs.cli.download_tutorials_data.unzip_file')
    @patch('model2obs.cli.download_tutorials_data.download_file')
    def test_main_destination_trailing_slash(self, mock_download, mock_unzip, tmp_path, monkeypatch):
        """Test main handles destination with trailing slash."""
        destination = str(tmp_path / "dest") + "/"
        
        test_args = [
            'prog',
            '--destination', destination
        ]
        monkeypatch.setattr('sys.argv', test_args)
        
        with patch('os.makedirs'):
            with patch('os.remove'):
                metadata_url = "https://zenodo.org/api/records/19204393"
                mock_response = {
                    'files': [
                        {
                            'key': 'file.zip',
                            'links': {'self': 'https://zenodo.org/api/files/1/file.zip'}
                        }
                    ]
                }
                
                responses.add(
                    responses.GET,
                    metadata_url,
                    json=mock_response,
                    status=200
                )
                
                download_tutorials_data.main()
                
                assert mock_download.called
    
    @responses.activate
    @patch('model2obs.cli.download_tutorials_data.unzip_file')
    @patch('model2obs.cli.download_tutorials_data.download_file')
    def test_main_non_zip_files_not_extracted(self, mock_download, mock_unzip, tmp_path, monkeypatch, capsys):
        """Test main only extracts ZIP files."""
        test_args = ['prog']
        monkeypatch.setattr('sys.argv', test_args)
        
        with patch('os.makedirs'):
            metadata_url = "https://zenodo.org/api/records/19204393"
            mock_response = {
                'files': [
                    {
                        'key': 'data.txt',
                        'links': {'self': 'https://zenodo.org/api/files/1/data.txt'}
                    }
                ]
            }
            
            responses.add(
                responses.GET,
                metadata_url,
                json=mock_response,
                status=200
            )
            
            download_tutorials_data.main()
            
            assert not mock_unzip.called
            
            captured = capsys.readouterr()
            assert "Download complete" in captured.out
    
    @responses.activate
    @patch('model2obs.cli.download_tutorials_data.unzip_file')
    @patch('model2obs.cli.download_tutorials_data.download_file')
    def test_main_cleans_up_zip_files(self, mock_download, mock_unzip, tmp_path, monkeypatch, capsys):
        """Test main removes ZIP files after extraction."""
        test_args = ['prog']
        monkeypatch.setattr('sys.argv', test_args)
        
        with patch('os.makedirs'):
            with patch('os.remove') as mock_remove:
                metadata_url = "https://zenodo.org/api/records/19204393"
                mock_response = {
                    'files': [
                        {
                            'key': 'archive.zip',
                            'links': {'self': 'https://zenodo.org/api/files/1/archive.zip'}
                        }
                    ]
                }
                
                responses.add(
                    responses.GET,
                    metadata_url,
                    json=mock_response,
                    status=200
                )
                
                download_tutorials_data.main()
                
                captured = capsys.readouterr()
                assert "Removing zip archive" in captured.out
                assert mock_remove.called
