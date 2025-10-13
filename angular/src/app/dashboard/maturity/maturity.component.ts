import { Component } from '@angular/core';
import { LoginService } from '../../services/login.service';
import { HttpClient } from '@angular/common/http';

@Component({
  selector: 'app-maturity',
  standalone: false,
  templateUrl: './maturity.component.html',
  styleUrl: './maturity.component.scss'
})
export class MaturityComponent {
  
  selectedFile: File | null = null;
  results: any = null;
  detections = [];
  isLoading = false;
  constructor(
    private http: HttpClient,
  ) {}

  onFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files.length > 0) {
      this.selectedFile = input.files[0];
    }
  }

  uploadImage(): void {
    if (!this.selectedFile) return;
    this.isLoading = true;
    const formData = new FormData();
    formData.append('image', this.selectedFile);

    this.http.post('http://localhost:5000/upload', formData).subscribe({
      next: (response: any) => {
        console.log(response)
        this.results = response;
        this.detections = response.detections;
        this.isLoading = false;
      },
      error: (error:any) => {
        this.isLoading = false;
        console.error('Error uploading image:', error);
      }
    });
  }


}
