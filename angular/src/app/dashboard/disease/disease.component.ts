import { HttpClient } from '@angular/common/http';
import { Component } from '@angular/core';

@Component({
  selector: 'app-disease',
  standalone: false,
  templateUrl: './disease.component.html',
  styleUrl: './disease.component.scss'
})
export class DiseaseComponent {

  selectedFile: File | null = null;
  results: any = null;
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

    this.http.post('http://localhost:5000/disease', formData).subscribe({
      next: (response: any) => {
        console.log(response)
        this.results = response;
        this.isLoading = false;
      },
      error: (error:any) => {
        this.isLoading = false;
        console.error('Error uploading image:', error);
      }
    });
  }

}
