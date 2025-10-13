import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class PriceService {

  private apiUrl = 'http://127.0.0.1:5000/pepper-price';

  constructor(private http: HttpClient) {}

  getPrediction(params: any): Observable<any> {
    return this.http.post(this.apiUrl, params);
  }
}
