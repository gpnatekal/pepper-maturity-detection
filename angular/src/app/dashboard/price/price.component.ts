import { HttpClient } from '@angular/common/http';
import { Component } from '@angular/core';
import { FormBuilder, FormGroup, Validators } from '@angular/forms';
import { PriceService } from './service/price.service';

@Component({
  selector: 'app-price',
  standalone: false,
  templateUrl: './price.component.html',
  styleUrl: './price.component.scss'
})
export class PriceComponent {
 predictionForm!: FormGroup;
  result: any;
  loading = false;

  constructor(private fb: FormBuilder, private service: PriceService) {}

  ngOnInit() {
    this.predictionForm = this.fb.group({
      dateFrom: ['2018-01-01', Validators.required],
      dateTo: [new Date().toISOString().slice(0, 10), Validators.required],
      csvPath: [''],
      horizon: [7, [Validators.required, Validators.min(1)]],
      seqLen: [30, [Validators.required, Validators.min(1)]],
      marketFilter: [''],
      epochs: [40, [Validators.required, Validators.min(1)]],
      batchSize: [32, [Validators.required, Validators.min(1)]],
      testSplit: [0.2, [Validators.required, Validators.min(0.1), Validators.max(0.9)]],
      minDays: [200, [Validators.required, Validators.min(50)]]
    });
  }

  onSubmit() {
    if (this.predictionForm.invalid) return;

    this.loading = true;
    const params = this.predictionForm.value;

    this.service.getPrediction(params).subscribe({
      next: (res) => {
        this.result = res;
        this.loading = false;
      },
      error: (err) => {
        console.error(err);
        this.loading = false;
      }
    });
  }
}
