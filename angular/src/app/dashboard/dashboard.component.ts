import { Component } from '@angular/core';
import { LoginService } from '../services/login.service';
import { Router } from '@angular/router';

@Component({
  selector: 'app-dashboard',
  standalone: false,
  templateUrl: './dashboard.component.html',
  styleUrl: './dashboard.component.scss'
})
export class DashboardComponent {
  constructor(
    private loginService: LoginService,
    private router: Router
  ) {

  }
  logout(): void {
    this.loginService.googlesignOut().then(() => {
      this.router.navigate(['/login']);   // 👈 redirect here
    });
  }
}
